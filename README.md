<div align="center">

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" height="200px" width="400px" viewBox="0 0 400 100">
  <defs>
    <linearGradient x1="87.9681034%" y1="50%" x2="1.27351722%" y2="50%" id="linearGradient-1">
      <stop offset="0%" stop-color="#174889"></stop>
      <stop offset="67.6877392%" stop-color="#7c46ce"></stop>
      <stop offset="100%" stop-color="#c1c8ca"></stop>
    </linearGradient>
  </defs>
  <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd" font-family="Arial-BoldMT, Arial" font-weight="bold">
    <g id="gh-banner">
      <text id="gh-title-reflection" fill="url(#linearGradient-1)" font-size="72">
        <tspan x="200" y="120" text-anchor="middle">viteo</tspan>
      </text>
    </g>
  </g>
</svg>

**High-performance video frame extraction for Apple Silicon**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg)](https://www.apple.com/mac/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

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

> **Note:** PyPI package coming very soon!

### From source

```bash
git clone https://github.com/codeSamuraii/viteo
cd viteo
pip install -v -e .
```

#### Rebuilding

```bash
rm -rf dist/
pip install -e . --force-reinstall --no-deps
```

### Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.8+
- MLX framework
