# viteo

High-performance video frame extraction for Apple Silicon using AVFoundation/VideoToolbox with MLX integration.

## Usage

```python
import viteo

# Simple iteration
with viteo.open("video.mp4") as frames:
    for frame in frames:
        # frame is MLX array (height, width, 4) BGRA format
        process(frame)

# Or direct extraction
extractor = viteo.FrameExtractor("video.mp4")
for frame in extractor:
    process(frame)
```

## Development

Building/rebuilding the package in editable mode:
```bash
rm -rf dist/
pip install -e . --force-reinstall --no-deps
```

## Key Features

- **Hardware acelerated**: Zero-copy extraction using VideoToolbox with Metal compatibility
- **MLX native**: Direct integration with MLX arrays for GPU-ready processing
- **Optimized bindings**: `nanobind` extension with internal batching and GIL release for maximum throughput

## Architecture

The extension implements a three-layer architecture optimized for performance:

**C++ Core** (`frame_extractor.h/mm`)
- Minimal interface with only essential operations (open, extract_batch, reset)
- Direct CVPixelBuffer to memory copy with fast-path optimization
- Cached video properties and frame-level seeking support

**Objective-C++ Backend**
- AVFoundation/VideoToolbox integration with hardware acceleration
- IOSurface backing and Metal compatibility for GPU transfers
- Native BGRA format to avoid color conversion overhead

**Python Bindings** (`bindings.cpp`)
- Custom iterator using nanobind for minimal overhead
- Direct buffer protocol integration with MLX arrays
- Automatic batch management transparent to users

## License

MIT