"""
Device abstraction layer for MPS/CUDA compatibility.
Ensures float32 precision for MPS compatibility.
"""
import torch
import os


def get_device():
    """Get the best available device with MPS preference over CUDA for Apple Silicon."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_device_with_id(device_id=None):
    """Get device with optional ID (only applies to CUDA)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f"cuda:{device_id}")
        else:
            return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_device(device_id=None):
    """Set the current device (only applies to CUDA)."""
    if torch.cuda.is_available() and device_id is not None:
        torch.cuda.set_device(device_id)


def get_current_device():
    """Get current device ID (returns 0 for MPS/CPU)."""
    if torch.backends.mps.is_available():
        return 0
    elif torch.cuda.is_available():
        return torch.cuda.current_device()
    else:
        return 0


def get_device_name(device_id=0):
    """Get device name."""
    if torch.backends.mps.is_available():
        return "apple_silicon_mps"
    elif torch.cuda.is_available():
        return torch.cuda.get_device_name(device_id)
    else:
        return "cpu"


def empty_cache():
    """Empty device cache."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize():
    """Synchronize device operations."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def get_memory_info():
    """Get memory information (returns tuple of (free, total) in bytes)."""
    if torch.backends.mps.is_available():
        # MPS doesn't have direct memory query, return approximation
        try:
            allocated = torch.mps.current_allocated_memory()
            # Estimate based on typical Mac memory configurations
            total_estimate = 32 * 1024**3  # 32GB estimate
            free_estimate = max(0, total_estimate - allocated)
            return (free_estimate, total_estimate)
        except:
            return (16 * 1024**3, 32 * 1024**3)  # Fallback estimates
    elif torch.cuda.is_available():
        return torch.cuda.mem_get_info()
    else:
        return (8 * 1024**3, 16 * 1024**3)  # CPU fallback


def get_memory_stats():
    """Get detailed memory statistics."""
    if torch.backends.mps.is_available():
        try:
            allocated = torch.mps.current_allocated_memory()
            return {
                'allocated_bytes.all.current': allocated,
                'reserved_bytes.all.current': allocated,  # Approximation
                'active_bytes.all.current': allocated,
            }
        except:
            return {
                'allocated_bytes.all.current': 0,
                'reserved_bytes.all.current': 0,
                'active_bytes.all.current': 0,
            }
    elif torch.cuda.is_available():
        return torch.cuda.memory_stats()
    else:
        return {}


def is_mps():
    """Check if MPS is the current backend."""
    return torch.backends.mps.is_available()


def is_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def ensure_float32(tensor):
    """Ensure tensor is float32 for MPS compatibility."""
    if tensor.dtype in [torch.float16, torch.bfloat16]:
        return tensor.float()
    return tensor


def convert_model_to_float32(model):
    """Recursively convert all model parameters and buffers to float32."""
    if is_mps():
        for name, param in model.named_parameters():
            if param.dtype in [torch.float16, torch.bfloat16, torch.float64]:
                param.data = param.data.float()
        for name, buffer in model.named_buffers():
            if buffer.dtype in [torch.float16, torch.bfloat16, torch.float64]:
                buffer.data = buffer.data.float()
    return model


def ensure_mps_compatible_dtype(tensor):
    """Ensure tensor has MPS-compatible dtype (float32 instead of float64)."""
    if is_mps() and tensor.dtype == torch.float64:
        return tensor.float()
    elif tensor.dtype in [torch.float16, torch.bfloat16]:
        return tensor.float() if is_mps() else tensor
    return tensor


def configure_device_settings():
    """Configure device-specific settings for optimal performance."""
    if torch.cuda.is_available():
        # CUDA optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        # MPS-specific settings
        # Force float32 for better compatibility
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        # Try to prevent automatic promotion to float64
        try:
            torch.set_default_dtype(torch.float32)
        except:
            pass


class DeviceEvent:
    """Cross-platform event for timing operations."""
    
    def __init__(self, enable_timing=True):
        self.enable_timing = enable_timing
        self.start_time = None
        self.end_time = None
        
        if torch.cuda.is_available():
            self.cuda_event_start = torch.cuda.Event(enable_timing=enable_timing) if enable_timing else None
            self.cuda_event_end = torch.cuda.Event(enable_timing=enable_timing) if enable_timing else None
        else:
            self.cuda_event_start = None
            self.cuda_event_end = None
    
    def record(self):
        """Record event."""
        if torch.cuda.is_available() and self.cuda_event_start:
            self.cuda_event_start.record()
        elif self.enable_timing:
            import time
            self.start_time = time.time()
    
    def record_end(self):
        """Record end event."""
        if torch.cuda.is_available() and self.cuda_event_end:
            self.cuda_event_end.record()
        elif self.enable_timing:
            import time
            self.end_time = time.time()
    
    def elapsed_time(self, end_event):
        """Get elapsed time in milliseconds."""
        if torch.cuda.is_available() and self.cuda_event_start and end_event.cuda_event_end:
            return self.cuda_event_start.elapsed_time(end_event.cuda_event_end)
        elif self.start_time and end_event.end_time:
            return (end_event.end_time - self.start_time) * 1000  # Convert to ms
        else:
            return 0.0


def create_event(enable_timing=True):
    """Create a device event for timing."""
    return DeviceEvent(enable_timing=enable_timing)