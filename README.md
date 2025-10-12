<div align="center">

![logo](https://github.com/user-attachments/assets/a7e90f28-64db-4401-94de-f8b573d3eec8)

**High-performance video frame extraction for Apple Silicon**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/viteo.svg)](https://pypi.org/project/viteo/)
[![Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg)](https://www.apple.com/mac/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

----

- **Hardware-accelerated** extraction using AVFoundation/VideoToolbox
- **MLX native** with direct BGRA frame copy to `mx.array`
- **Lightweight bindings** using `nanobind` and GIL release
- **Pythonic** interface with simple iterators and context managers

<br />

## Getting started

```python
import viteo

# Simple iteration with context manager
with viteo.open("video.mp4") as frames:
    for frame in frames:
        # frame is MLX array (height, width, 4) BGRA format
        process(frame)

# Direct extraction
extractor = viteo.FrameExtractor("video.mp4")
for frame in extractor:
    process(frame)
```

## Installation

```bash
pip install viteo
```

### From source

```bash
git clone https://github.com/codeSamuraii/viteo
cd viteo
pip install -v -e .
```

To rebuild after changes:
```bash
pip install -e . --force-reinstall --no-deps
```

## Configuration
### Logging

You can enable debug logging with the `VITEO_DEBUG` environment variable:
```bash
$ VITEO_DEBUG=1 python example.py video_1080p.mp4

[viteo] Closed video resources
[viteo] Loaded asset from: tests/test-data/video_1080p.mp4
[viteo] Found 1 video track(s)
[viteo] Video metadata: 1920x1080 @ 23.976 fps, 267 total frames
[viteo] Allocated batch buffer for 16 frames
[viteo] Created track output with hardware acceleration
[viteo] Reader initialized successfully
  ...
```

### Batch size

Internally, `viteo` passes frames to Python in batches for performance.
The default batch size is 8 frames, but you can change it by passing the `batch_size` argument:
```python
# Values between 2 and 16 are optimal
with viteo.open("video.mp4", batch_size=2) as frames:
    for frame in frames:
        process(frame)
```
