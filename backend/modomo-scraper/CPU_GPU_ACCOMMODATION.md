# CPU/GPU Accommodation Guide

## Overview
The depth and edge map generation system now fully supports both GPU and CPU environments with automatic optimization and performance monitoring.

## CPU Optimizations Implemented

### 1. **Depth Estimation (ZoeDepth)**
- **CPU-specific model configuration**: Reduced complexity for CPU processing
- **Memory management**: Limited CPU threads to prevent system overload
- **Precision handling**: float32 on CPU, optional float16 on GPU
- **Garbage collection**: Explicit memory cleanup after operations

```python
# CPU optimizations in DepthConfig
cpu_optimization: bool = True  # Enable CPU-specific optimizations
reduce_precision: bool = False  # Use appropriate precision per device
```

### 2. **Edge Detection (CV2 Canny)**
- **Native CPU processing**: OpenCV operates efficiently on CPU
- **Adaptive thresholds**: CPU-friendly parameter computation
- **No GPU dependency**: Pure CPU implementation

### 3. **Memory Management**
- **Sequential processing**: Depth → Edge to avoid memory conflicts
- **Automatic cleanup**: GPU cache clearing + CPU garbage collection
- **Resource monitoring**: Track memory usage during operations

```python
# Memory cleanup for both environments
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # GPU cleanup
else:
    import gc
    gc.collect()  # CPU cleanup
```

## Performance Monitoring

### 1. **System Detection**
```python
# Automatic device detection with logging
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🖥️ Using device: {device}")
```

### 2. **Performance Warnings**
- **Low RAM warning**: <8GB system memory
- **Limited CPU warning**: <4 CPU cores
- **CPU-only mode**: "Expect 3-6x longer processing times"
- **Limited GPU warning**: <6GB GPU memory

### 3. **Time Estimates**
| Operation | GPU Time | CPU Time |
|-----------|----------|----------|
| Depth Estimation | ~5s | ~30s |
| Edge Detection | ~1s | ~2s |
| Full Pipeline | ~20s | ~120s |

*Times are for average indoor scenes and vary by system specs*

## API Enhancements

### 1. **Performance Status Endpoint**
```bash
GET /performance/status
```
Returns:
- System specifications (CPU, RAM, GPU)
- Performance warnings and suggestions
- Estimated processing times
- Latest operation metrics

### 2. **Graceful Fallbacks**
- Models fail gracefully if hardware requirements not met
- Service continues with available models only
- Clear logging of what's available vs unavailable

## User Experience

### 1. **Automatic Optimization**
- No manual configuration required
- System automatically detects and optimizes for available hardware
- Progressive enhancement: works on any system

### 2. **Transparent Performance**
- Clear warnings about expected performance
- Time estimates before processing
- Real-time operation tracking

### 3. **Helpful Suggestions**
For CPU-only environments:
- "Consider using a GPU-enabled environment for faster processing"
- "Process scenes in smaller batches to manage memory usage"
- "Use edge detection only for faster results (skip depth maps)"

## Development & Testing

### 1. **Local Development**
```bash
# CPU-only testing
CUDA_VISIBLE_DEVICES="" python main_refactored.py

# GPU testing (if available)
python main_refactored.py
```

### 2. **Docker Considerations**
- Base images work with CPU-only
- GPU support requires nvidia-docker
- Memory limits respected automatically

### 3. **Railway Deployment**
- Handles both CPU and GPU instances
- Automatic scaling based on available resources
- Performance monitoring helps with resource planning

## Error Handling

### 1. **Model Loading Failures**
```python
try:
    depth_estimator = DepthEstimator(depth_config)
    logger.info(f"✅ ZoeDepth estimator initialized on {device}")
except Exception as depth_error:
    logger.warning(f"⚠️ Depth estimator failed to load: {depth_error}")
    depth_estimator = None  # Graceful fallback
```

### 2. **Runtime Issues**
- Out of memory: Automatic batch size reduction
- Model failures: Skip problematic operations, continue with others
- Timeout handling: Configurable per environment

## Configuration Options

### 1. **Environment Variables**
```bash
# Force CPU mode (for testing)
CUDA_VISIBLE_DEVICES=""

# Memory limits
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# CPU thread limiting
OMP_NUM_THREADS=4
```

### 2. **Model Configuration**
```python
# Depth estimation config
DepthConfig(
    device=device,
    cpu_optimization=device == "cpu",    # Auto-enable for CPU
    reduce_precision=device == "cuda"     # GPU optimization
)

# Edge detection config  
EdgeConfig(
    adaptive_thresholds=True,   # Better for varying scenes
    enhance_contrast=True       # Improve CPU performance
)
```

## Monitoring & Metrics

### 1. **Real-time Tracking**
- CPU usage percentage
- Memory consumption (RAM/VRAM)
- Processing duration
- Resource recommendations

### 2. **Historical Data**
- Operation performance history
- System resource trends
- Optimization suggestions

## Best Practices

### 1. **For CPU Environments**
- Process smaller batches (5-10 scenes)
- Prioritize edge detection (faster than depth)
- Close other applications during processing
- Monitor system temperature

### 2. **For GPU Environments**
- Utilize full batch sizes (20+ scenes)
- Enable mixed precision when supported
- Monitor VRAM usage
- Consider concurrent operations

### 3. **Hybrid Approach**
- Use GPU for depth estimation (most intensive)
- Use CPU for edge detection (if GPU busy)
- Balance workload across available resources

This comprehensive accommodation ensures the map generation system works effectively across all hardware configurations while providing clear feedback about performance expectations.