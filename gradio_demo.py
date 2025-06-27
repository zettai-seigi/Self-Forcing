"""
Enhanced Gradio Demo for Self-Forcing with MPS Support and Real-time Streaming
Combines the reliability of Gradio streaming with Apple Silicon optimizations
"""

import os
import re
import random
import argparse
import hashlib
import time
import uuid
from PIL import Image
import torch
import gradio as gr
from omegaconf import OmegaConf
import imageio
import av
import numpy as np

# Import our MPS-optimized modules
from pipeline import CausalInferencePipeline
from demo_utils.constant import ZERO_VAE_CACHE
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from utils.device import (
    get_device, configure_device_settings, is_mps, is_cuda, 
    convert_model_to_float32, get_memory_info
)

# Set MPS memory configuration BEFORE device configuration
if torch.backends.mps.is_available():
    import os
    # Clear all MPS-related environment variables that might cause conflicts
    mps_env_vars = [
        'PYTORCH_MPS_HIGH_WATERMARK_RATIO',
        'MPS_WATERMARK_RATIO', 
        'PYTORCH_MPS_LOW_WATERMARK_RATIO',
        'MPS_LOW_WATERMARK_RATIO'
    ]
    for var in mps_env_vars:
        if var in os.environ:
            print(f"🧹 Clearing existing {var}={os.environ[var]}")
            del os.environ[var]
    
    # Don't set any watermark ratios - let MPS use defaults
    print("🍎 Using default MPS memory management to prevent ratio errors")

# Configure device settings for MPS/CUDA compatibility
configure_device_settings()
device = get_device()
print(f'🚀 Using device: {device}')

# Memory management based on device type
if is_mps():
    import psutil
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f'🍎 Running on Apple Silicon MPS - {total_memory_gb:.1f}GB total, {available_memory_gb:.1f}GB available')
    # Use conservative memory settings for MPS to prevent crashes
    low_memory = True  # Always use conservative settings on MPS
    print('⚠️ Using conservative memory settings to prevent system crashes on MPS')
elif is_cuda():
    from demo_utils.memory import get_cuda_free_memory_gb
    gpu_memory = get_cuda_free_memory_gb(device)
    print(f'🔥 CUDA device with {gpu_memory:.1f}GB free VRAM')
    low_memory = gpu_memory < 40
else:
    print('💻 Running on CPU')
    low_memory = True

# Argument parsing
parser = argparse.ArgumentParser(description="Enhanced Gradio Demo for Self-Forcing with MPS Support")
parser.add_argument('--port', type=int, default=7860, help="Port to run the Gradio app on.")
parser.add_argument('--host', type=str, default='0.0.0.0', help="Host to bind the Gradio app to.")
parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/self_forcing_dmd.pt', help="Path to the model checkpoint.")
parser.add_argument("--config_path", type=str, default='./configs/self_forcing_dmd.yaml', help="Path to the model config.")
parser.add_argument('--share', action='store_true', help="Create a public Gradio link.")
parser.add_argument('--trt', action='store_true', help="Use TensorRT optimized VAE decoder.")
parser.add_argument('--fps', type=float, default=15.0, help="Playback FPS for frame streaming.")
args = parser.parse_args()

# Disable TensorRT on MPS
if is_mps() and args.trt:
    print("⚠️ TensorRT is not supported on MPS. Disabling TensorRT.")
    args.trt = False

try:
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
except FileNotFoundError as e:
    print(f"❌ Error loading config file: {e}")
    print("Please ensure config files are in the correct path.")
    exit(1)

# Initialize Models with MPS optimizations
print("🔧 Initializing models...")
text_encoder = WanTextEncoder()
transformer = WanDiffusionWrapper(is_causal=True)

try:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    transformer.load_state_dict(state_dict.get('generator_ema', state_dict.get('generator')))
    print("✅ Model checkpoint loaded successfully")
except FileNotFoundError as e:
    print(f"❌ Error loading checkpoint: {e}")
    print(f"Please ensure the checkpoint '{args.checkpoint_path}' exists.")
    exit(1)

# Apply device-specific optimizations
if is_mps():
    print("🍎 Applying MPS optimizations...")
    # Use float32 for MPS compatibility
    text_encoder = text_encoder.eval().to(dtype=torch.float32).requires_grad_(False)
    transformer = transformer.eval().to(dtype=torch.float32).requires_grad_(False)
    
    # Deep conversion of all parameters and buffers
    text_encoder = convert_model_to_float32(text_encoder)
    transformer = convert_model_to_float32(transformer)
else:
    # Use float16 for CUDA
    text_encoder = text_encoder.eval().to(dtype=torch.float16).requires_grad_(False)
    transformer = transformer.eval().to(dtype=torch.float16).requires_grad_(False)

# Move models to device
text_encoder.to(device)
transformer.to(device)

# Initialize VAE decoder - same as demo.py
if args.trt and not is_mps():
    print("🚀 Initializing TensorRT VAE decoder...")
    from demo_utils.vae import VAETRTWrapper
    vae_decoder = VAETRTWrapper()
else:
    print("🎨 Initializing default VAE decoder...")
    vae_decoder = VAEDecoderWrapper()
    
    # Load VAE weights - THIS WAS MISSING!
    print("📦 Loading VAE weights...")
    vae_state_dict = torch.load('wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth', map_location="cpu")
    decoder_state_dict = {}
    for key, value in vae_state_dict.items():
        if 'decoder.' in key or 'conv2' in key:
            decoder_state_dict[key] = value
    vae_decoder.load_state_dict(decoder_state_dict)
    print("✅ VAE weights loaded successfully")

# Configure VAE decoder properly
vae_decoder.eval()
vae_decoder.requires_grad_(False)

# Ensure VAE decoder is properly configured for device
if is_mps():
    vae_decoder = vae_decoder.to(dtype=torch.float32)
    vae_decoder = convert_model_to_float32(vae_decoder)
    print("🍎 VAE decoder converted to float32 for MPS compatibility")
else:
    vae_decoder = vae_decoder.to(dtype=torch.float16)

vae_decoder.to(device)

print("✅ All models initialized successfully")

# Global state management
# Determine device type
if is_mps():
    device_type = "mps"
    memory_gb = total_memory_gb
elif is_cuda():
    device_type = "cuda"
    memory_gb = gpu_memory
else:
    device_type = "cpu"
    memory_gb = 8

APP_STATE = {
    "torch_compile_applied": False,
    "fp8_applied": False,
    "current_use_taehv": False,
    "current_vae_decoder": vae_decoder,
    "device_type": device_type,
    "total_memory_gb": memory_gb,
    "low_memory_mode": low_memory
}

def get_device_info():
    """Get current device information for display"""
    if is_mps():
        import psutil
        available = psutil.virtual_memory().available / (1024**3)
        total = psutil.virtual_memory().total / (1024**3)
        return f"🍎 Apple Silicon MPS | {total:.1f}GB total | {available:.1f}GB available"
    elif is_cuda():
        try:
            from demo_utils.memory import get_cuda_free_memory_gb
            free_vram = get_cuda_free_memory_gb(device)
            return f"🔥 NVIDIA CUDA | {free_vram:.1f}GB VRAM free"
        except ImportError:
            return "🔥 NVIDIA CUDA | Memory info unavailable"
    else:
        return "💻 CPU | No GPU acceleration"

def frames_to_ts_file(frames, output_path, fps=15.0):
    """Convert frames to MPEG-TS file for streaming"""
    try:
        container = av.open(output_path, mode='w', format='mpegts')
        stream = container.add_stream('h264', rate=fps)
        stream.width = frames[0].shape[1]
        stream.height = frames[0].shape[0]
        stream.pix_fmt = 'yuv420p'
        
        for frame_array in frames:
            # Ensure frame is in correct format (HWC, uint8)
            if frame_array.dtype != np.uint8:
                frame_array = (frame_array * 255).astype(np.uint8)
            
            if len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
                # RGB to YUV420p conversion
                frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
                frame = frame.reformat(format='yuv420p')
                
                for packet in stream.encode(frame):
                    container.mux(packet)
        
        # Flush remaining packets
        for packet in stream.encode():
            container.mux(packet)
        
        container.close()
        return True
    except Exception as e:
        print(f"❌ Error creating TS file: {e}")
        return False

def _validate_inputs(prompt):
    """Validate input parameters"""
    if not prompt.strip():
        return False, "❌ Please enter a prompt"
    return True, ""

def _setup_pipeline():
    """Initialize the inference pipeline"""
    return CausalInferencePipeline(
        config, device=device, generator=transformer, text_encoder=text_encoder, 
        vae=APP_STATE["current_vae_decoder"]
    )

def _handle_seed(seed):
    """Handle seed generation"""
    if seed == -1:
        seed = random.randint(0, 2**32)
    print(f"🎬 Starting generation with seed: {seed}")
    return seed

def _apply_torch_compile(enable_torch_compile):
    """Apply torch.compile optimization"""
    if enable_torch_compile and not is_mps():
        if not APP_STATE["torch_compile_applied"]:
            print("🔥 Applying torch.compile...")
            transformer.compile(mode="max-autotune-no-cudagraphs")
            APP_STATE["torch_compile_applied"] = True
        return True, ""
    elif enable_torch_compile and is_mps():
        return False, "⚠️ torch.compile is not supported on MPS"
    return True, ""

def _apply_fp8_quantization(enable_fp8):
    """Apply FP8 quantization optimization"""
    if enable_fp8 and not is_mps():
        if not APP_STATE["fp8_applied"]:
            print("⚡ Applying FP8 quantization...")
            try:
                from torchao.quantization.quant_api import quantize_, Float8DynamicActivationFloat8WeightConfig, PerTensor
                quantize_(transformer, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
                APP_STATE["fp8_applied"] = True
                return True, ""
            except Exception as e:
                return False, f"❌ FP8 quantization failed: {e}"
    elif enable_fp8 and is_mps():
        return False, "⚠️ FP8 quantization is not supported on MPS"
    return True, ""

def _create_taehv_wrapper():
    """Create TAEHV wrapper class"""
    from demo_utils.taehv import TAEHV
    from omegaconf import OmegaConf
    
    class TAEHVDiffusersWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dtype = torch.float32 if is_mps() else torch.float16
            taehv_checkpoint_path = "checkpoints/taew2_1.pth"
            
            # Download TAEHV checkpoint if not exists
            if not os.path.exists(taehv_checkpoint_path):
                print("📥 Downloading TAEHV checkpoint...")
                os.makedirs("checkpoints", exist_ok=True)
                download_url = "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth"
                import urllib.request
                urllib.request.urlretrieve(download_url, taehv_checkpoint_path)
                print(f"✅ Successfully downloaded taew2_1.pth to {taehv_checkpoint_path}")
            
            self.taehv = TAEHV(checkpoint_path=taehv_checkpoint_path).to(self.dtype)
            self.config = OmegaConf.create({"scaling_factor": 1.0})

        def decode(self, latents, return_dict=None):
            return self.taehv.decode_video(latents, parallel=False).mul_(2).sub_(1)
            
        def __call__(self, latents, *args):
            return self.decode(latents)
    
    return TAEHVDiffusersWrapper()

def _switch_vae_decoder(use_taehv, pipeline):
    """Switch VAE decoder between TAEHV and default"""
    if use_taehv != APP_STATE["current_use_taehv"]:
        print(f"🔄 Switching VAE decoder to {'TAEHV' if use_taehv else 'default VAE'}")
        
        if use_taehv:
            try:
                APP_STATE["current_vae_decoder"] = _create_taehv_wrapper()
                
                # Configure for device
                if is_mps():
                    APP_STATE["current_vae_decoder"] = convert_model_to_float32(APP_STATE["current_vae_decoder"])
                
                APP_STATE["current_vae_decoder"].to(device)
                APP_STATE["current_vae_decoder"].eval()
                APP_STATE["current_vae_decoder"].requires_grad_(False)
                
            except Exception as e:
                return False, f"❌ Failed to initialize TAEHV: {e}"
        else:
            APP_STATE["current_vae_decoder"] = vae_decoder
        
        APP_STATE["current_use_taehv"] = use_taehv
        pipeline.vae = APP_STATE["current_vae_decoder"]
    
    return True, ""

def _encode_text_prompt(prompt):
    """Encode text prompt and convert to appropriate dtype"""
    conditional_dict = text_encoder(text_prompts=[prompt])
    
    # Use device-appropriate dtype
    if is_mps():
        for key, value in conditional_dict.items():
            conditional_dict[key] = value.float()
    else:
        for key, value in conditional_dict.items():
            conditional_dict[key] = value.to(dtype=torch.float16)
    
    return conditional_dict

def _setup_generation_parameters():
    """Setup generation parameters and caches"""
    shape = [1, 21, 16, 60, 104]  # [batch, channels, temporal, height, width]
    num_blocks = 7  # Each block generates 3 frames
    
    # Create noise
    noise = torch.randn(shape, device=device, dtype=torch.float32 if is_mps() else torch.float16)
    
    # Initialize pipeline caches
    generation_dtype = torch.float32 if is_mps() else torch.float16
    
    return shape, num_blocks, noise, generation_dtype

def _initialize_vae_cache(use_taehv):
    """Initialize VAE cache based on decoder type"""
    if use_taehv:
        vae_cache = None
    else:
        vae_cache = list(ZERO_VAE_CACHE)  # Make a copy
        for i in range(len(vae_cache)):
            if is_mps():
                vae_cache[i] = vae_cache[i].to(device=device, dtype=torch.float32)
            else:
                vae_cache[i] = vae_cache[i].to(device=device, dtype=torch.float16)
    return vae_cache

def _create_status_html(idx, num_blocks, block_denoise_time, decode_time, total_frames_yielded):
    """Create status HTML for progress display"""
    progress = (idx + 1) / num_blocks * 100
    return f"""
    <div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
        <h3 style='margin: 0 0 10px 0;'>🎬 Block {idx+1}/{num_blocks} Complete</h3>
        <div style='background: rgba(255,255,255,0.2); border-radius: 10px; padding: 8px; margin: 5px 0;'>
            <div style='background: linear-gradient(90deg, #ff6b6b, #4ecdc4); height: 8px; border-radius: 4px; width: {progress}%; transition: width 0.3s ease;'></div>
        </div>
        <p style='margin: 5px 0;'>⚡ Denoising: {block_denoise_time:.1f}s | 🎨 Decoding: {decode_time:.1f}s</p>
        <p style='margin: 5px 0; font-size: 0.9em;'>📊 {total_frames_yielded} frames generated</p>
    </div>
    """

def _create_final_status_html(total_frames_yielded, generation_time):
    """Create final completion status HTML"""
    return f"""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); border-radius: 15px; color: white;'>
        <h2 style='margin: 0 0 15px 0;'>🎉 Generation Complete!</h2>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin: 15px 0;'>
            <div style='background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px;'>
                <div style='font-size: 1.5em; font-weight: bold;'>{total_frames_yielded}</div>
                <div style='font-size: 0.9em;'>Total Frames</div>
            </div>
            <div style='background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px;'>
                <div style='font-size: 1.5em; font-weight: bold;'>{generation_time:.1f}s</div>
                <div style='font-size: 0.9em;'>Generation Time</div>
            </div>
            <div style='background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px;'>
                <div style='font-size: 1.5em; font-weight: bold;'>{(total_frames_yielded/generation_time):.1f}</div>
                <div style='font-size: 0.9em;'>Frames/Second</div>
            </div>
        </div>
        <p style='margin: 10px 0;'>✨ Video ready for playback!</p>
    </div>
    """

def _process_single_block(pipeline, noise, conditional_dict, current_start_frame, current_num_frames, idx):
    """Process a single generation block"""
    block_start_time = time.time()
    
    noisy_input = noise[:, current_start_frame : current_start_frame + current_num_frames]
    
    # Denoising steps
    for step_idx, current_timestep in enumerate(pipeline.denoising_step_list):
        timestep = torch.ones([1, current_num_frames], device=noise.device, 
                            dtype=torch.int64) * current_timestep
        
        _, denoised_pred = pipeline.generator(
            noisy_image_or_video=noisy_input, conditional_dict=conditional_dict,
            timestep=timestep, kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start_frame * pipeline.frame_seq_length
        )
        
        if step_idx < len(pipeline.denoising_step_list) - 1:
            next_timestep = pipeline.denoising_step_list[step_idx + 1]
            next_timestep_tensor = torch.ones([1, current_num_frames], 
                                            device=noise.device, dtype=torch.int64) * next_timestep
            noisy_input = pipeline.scheduler.add_noise(
                denoised_pred, torch.randn_like(denoised_pred), next_timestep_tensor
            )
    
    block_denoise_time = time.time() - block_start_time
    print(f"⚡ Block {idx+1} denoising completed in {block_denoise_time:.2f}s")
    
    return denoised_pred, block_denoise_time

def _decode_vae_block(pipeline, denoised_pred, use_taehv, vae_cache, idx):
    """Decode latents to pixels using VAE"""
    decode_start = time.time()
    
    # VAE decoding using same logic as demo.py
    if use_taehv:
        pixels = pipeline.vae.decode(denoised_pred)
    else:
        if is_mps():
            vae_input = denoised_pred.float()
        else:
            vae_input = denoised_pred.half()
        pixels, vae_cache = pipeline.vae(vae_input, *vae_cache)
    
    # Skip first 3 frames of first block (noise) 
    if idx == 0:
        pixels = pixels[:, 3:, :, :, :]
    
    pixels = torch.clamp(pixels.float(), -1., 1.) * 127.5 + 127.5
    pixels = pixels.to(torch.uint8).cpu().numpy()
    
    decode_time = time.time() - decode_start
    print(f"🎨 Block {idx+1} VAE decoding completed in {decode_time:.2f}s")
    
    return pixels, vae_cache, decode_time

def _convert_pixels_to_frames(pixels):
    """Convert pixel tensor to list of frame arrays"""
    _, num_frames_in_block = pixels.shape[:2]
    all_frames_from_block = []
    
    for frame_idx in range(num_frames_in_block):
        frame = pixels[0, frame_idx]  # [H, W, C]
        # Transpose from CHW to HWC if needed
        if len(frame.shape) == 3 and frame.shape[0] in [3, 4]:  # Channels first
            frame = np.transpose(frame, (1, 2, 0))
        all_frames_from_block.append(frame)
    
    return all_frames_from_block

def _run_generation_loop(pipeline, noise, conditional_dict, all_num_frames, num_blocks, use_taehv, vae_cache, fps):
    """Run the main generation loop"""
    current_start_frame = 0
    total_frames_yielded = 0
    
    for idx, current_num_frames in enumerate(all_num_frames):
        yield None, f"🔄 Processing block {idx+1}/{num_blocks}..."
        
        # Process denoising for this block
        denoised_pred, block_denoise_time = _process_single_block(
            pipeline, noise, conditional_dict, current_start_frame, 
            current_num_frames, idx
        )
        
        # Update KV cache for next block (except for last block)
        if idx < len(all_num_frames) - 1:
            timestep = torch.zeros([1, current_num_frames], device=noise.device, dtype=torch.int64)
            pipeline.generator(
                noisy_image_or_video=denoised_pred, conditional_dict=conditional_dict,
                timestep=timestep, kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
            )
        
        # VAE decoding
        yield None, f"🎨 Decoding block {idx+1} to pixels..."
        pixels, vae_cache, decode_time = _decode_vae_block(
            pipeline, denoised_pred, use_taehv, vae_cache, idx
        )
        
        # Clear intermediate tensors
        del denoised_pred
        if is_mps():
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()
        
        # Convert to frames and create video chunk
        all_frames_from_block = _convert_pixels_to_frames(pixels)
        total_frames_yielded += len(all_frames_from_block)
        
        # Create and yield the video chunk with status
        try:
            import uuid
            chunk_uuid = str(uuid.uuid4())[:8]
            ts_filename = f"block_{idx:04d}_{chunk_uuid}.ts"
            ts_path = os.path.join("gradio_tmp", ts_filename)
            
            if frames_to_ts_file(all_frames_from_block, ts_path, fps):
                status_html = _create_status_html(idx, num_blocks, block_denoise_time, 
                                                decode_time, total_frames_yielded)
                yield ts_path, status_html
                
        except Exception as e:
            print(f"⚠️ Error encoding block {idx}: {e}")
            yield None, f"⚠️ Error encoding block {idx+1}: {e}"
        
        current_start_frame += current_num_frames
        
        # Clean up memory after each block
        del pixels
        if is_mps():
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()
    
    return total_frames_yielded

def generate_video_stream(prompt, seed=-1, fps=15.0, use_taehv=False, enable_torch_compile=False, enable_fp8=False):
    """Generate video with real-time streaming using yield"""
    
    # Validate inputs
    valid, error_msg = _validate_inputs(prompt)
    if not valid:
        yield None, error_msg
        return
    
    # Initialize components
    generation_start_time = time.time()
    pipeline = _setup_pipeline()
    seed = _handle_seed(seed)
    
    # Apply optimizations
    success, error_msg = _apply_torch_compile(enable_torch_compile)
    if not success:
        yield None, error_msg
        return
        
    success, error_msg = _apply_fp8_quantization(enable_fp8)
    if not success:
        yield None, error_msg
        return
    
    # Switch VAE decoder if needed
    try:
        success, error_msg = _switch_vae_decoder(use_taehv, pipeline)
        if not success:
            yield None, error_msg
            return
    except Exception as e:
        yield None, f"❌ Failed to switch VAE decoder: {e}"
        return
    
    # Start status update
    yield None, "🚀 Initializing generation..."
    
    try:
        # Text encoding
        yield None, "📝 Encoding text prompt..."
        conditional_dict = _encode_text_prompt(prompt)
        
        # Setup generation parameters
        yield None, "🎲 Generating noise..."
        _, num_blocks, noise, generation_dtype = _setup_generation_parameters()
        
        # Initialize pipeline caches
        pipeline._initialize_kv_cache(batch_size=1, dtype=generation_dtype, device=device)
        pipeline._initialize_crossattn_cache(batch_size=1, dtype=generation_dtype, device=device)
        
        # Initialize VAE cache
        vae_cache = _initialize_vae_cache(use_taehv)
        
        # Ensure pipeline VAE is properly configured
        if is_mps():
            pipeline.vae = pipeline.vae.to(device=device, dtype=torch.float32)
            pipeline.vae = convert_model_to_float32(pipeline.vae)
            print("🍎 Pipeline VAE ensured to be float32 on MPS")
        else:
            pipeline.vae = pipeline.vae.to(device=device, dtype=torch.float16)
        
        # Ensure temp directory exists
        os.makedirs("gradio_tmp", exist_ok=True)
        
        # Use same frame structure as demo.py
        all_num_frames = [pipeline.num_frame_per_block] * num_blocks
        
        # Run generation loop
        total_frames_yielded = 0
        generation_loop = _run_generation_loop(
            pipeline, noise, conditional_dict, all_num_frames, 
            num_blocks, use_taehv, vae_cache, fps
        )
        
        for result in generation_loop:
            if isinstance(result, int):  # Final return value
                total_frames_yielded = result
                break
            else:
                yield result  # Yield intermediate results
        
        # Final completion
        generation_time = time.time() - generation_start_time
        final_status = _create_final_status_html(total_frames_yielded, generation_time)
        yield None, final_status
        print(f"✅ Generation complete! {total_frames_yielded} frames in {generation_time:.2f}s")
        
    except Exception as e:
        yield None, f"❌ Generation failed: {str(e)}"
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()

# Enhanced Gradio UI
with gr.Blocks(
    title="Self-Forcing Video Generator", 
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .device-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .settings-group {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    """
) as demo:
    
    # Header
    gr.Markdown(
        """
        # 🚀 Self-Forcing Video Generator
        ### Real-time AI Video Generation with Apple Silicon MPS Support
        
        Advanced video generation with device-optimized performance and real-time streaming preview.
        """
    )
    
    # Device info display
    gr.HTML(
        f"""
        <div class="device-info">
            <h3 style="margin: 0 0 10px 0;">🖥️ System Information</h3>
            <p style="margin: 5px 0; font-size: 1.1em;">{get_device_info()}</p>
            <p style="margin: 5px 0; font-size: 0.9em;">
                {'🔒 Low Memory Mode' if low_memory else '⚡ Optimized Memory Mode'} | 
                {'🍎 MPS Optimizations Active' if is_mps() else '🔥 CUDA Optimizations Active'}
            </p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            # Main controls
            with gr.Group():
                prompt = gr.Textbox(
                    label="📝 Prompt", 
                    placeholder="A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage...", 
                    lines=4,
                    value=""
                )
            
            with gr.Row():
                generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ Stop", variant="stop")
            
            # Examples
            gr.Markdown("### 🎯 Quick Examples")
            gr.Examples(
                examples=[
                    "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse.",
                    "A white and orange tabby cat is seen happily darting through a dense garden, chasing something with wide, happy eyes as it jogs forward through the narrow path between plants.",
                    "A close-up shot of a ceramic teacup slowly pouring water into a glass mug. The water flows smoothly creating gentle ripples as it fills up.",
                    "A playful cat playing an electronic guitar, strumming the strings with its front paws. The cat sits comfortably on a small stool in a cozy, dimly lit room."
                ],
                inputs=[prompt],
            )
            
            # Advanced settings
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    seed = gr.Number(
                        label="🎲 Seed", 
                        value=-1, 
                        info="Use -1 for random seed",
                        precision=0
                    )
                    fps = gr.Slider(
                        label="🎞️ Playback FPS", 
                        minimum=1, 
                        maximum=30, 
                        value=args.fps, 
                        step=1,
                        info="Frames per second for playback"
                    )
                
                with gr.Row():
                    # Only show compatible options based on device
                    if not is_mps():
                        enable_torch_compile = gr.Checkbox(
                            label="🔥 torch.compile",
                            info="Faster inference after compilation",
                            value=False
                        )
                        enable_fp8 = gr.Checkbox(
                            label="⚡ FP8 Quantization",
                            info="Reduces memory usage",
                            value=False
                        )
                    else:
                        enable_torch_compile = gr.Checkbox(
                            label="🔥 torch.compile (Not supported on MPS)",
                            info="Not available for Apple Silicon",
                            value=False,
                            interactive=False
                        )
                        enable_fp8 = gr.Checkbox(
                            label="⚡ FP8 Quantization (Not supported on MPS)",
                            info="Not available for Apple Silicon",
                            value=False,
                            interactive=False
                        )
                    
                    use_taehv = gr.Checkbox(
                        label="✨ TAEHV VAE",
                        info="Faster decoding, lower quality",
                        value=False
                    )
        
        with gr.Column(scale=3):
            # Video streaming display
            gr.Markdown("### 📺 Real-time Video Stream")
            
            streaming_video = gr.Video(
                label="Live Generation",
                streaming=True,
                loop=True,
                height=400,
                autoplay=True,
                show_label=False
            )
            
            status_display = gr.HTML(
                value="""
                <div style='text-align: center; padding: 20px; color: #666; border: 2px dashed #ddd; border-radius: 10px;'>
                    <h3>🎬 Ready to Generate</h3>
                    <p>Configure your prompt and settings, then click 'Generate Video' to start real-time streaming.</p>
                </div>
                """,
                label="Generation Status"
            )
    
    # Connect the generator to the streaming video
    generate_btn.click(
        fn=generate_video_stream,
        inputs=[prompt, seed, fps, use_taehv, enable_torch_compile, enable_fp8],
        outputs=[streaming_video, status_display]
    )

# Launch the enhanced demo
if __name__ == "__main__":
    # Clean up temp directory
    if os.path.exists("gradio_tmp"):
        import shutil
        shutil.rmtree("gradio_tmp")
    os.makedirs("gradio_tmp", exist_ok=True)
    
    print("🚀 Starting Enhanced Self-Forcing Demo with MPS Support")
    print("📁 Temporary files: gradio_tmp/")
    print("🎯 Video streaming: PyAV (MPEG-TS/H.264)")
    
    # Determine device display name
    if is_mps():
        device_display = 'MPS'
    elif is_cuda():
        device_display = 'CUDA'
    else:
        device_display = 'CPU'
    
    print(f"⚡ Device: {device} ({device_display})")
    
    demo.queue(max_size=10).launch(
        server_name=args.host, 
        server_port=args.port, 
        share=args.share,
        show_error=True,
        max_threads=40
    )