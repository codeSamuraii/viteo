# videoextractor

Python extension for hardware-accelerated video frame extraction using AVFoundation/VideoToolbox and `nanobind`.

## Features

- Hardware-accelerated frame extraction using Apple's VideoToolbox
- Stream frames in real time via callback or queue

## Installation

Let `poetry` handle the installation and building of the package:
```bash
poetry install -v --no-cache
```

Or build manually with CMake:

```bash
cd videoextractor
mkdir build && cd build
cmake ..
cmake --build . --config Debug

poetry run pip install -e .
```

## Architecture

The project uses:
- **Objective-C++** for interfacing with Apple frameworks (AVFoundation, VideoToolbox)
- **nanobind** for creating Python bindings with minimal overhead

## License

MIT
