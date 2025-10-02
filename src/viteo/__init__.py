"""
Hardware-accelerated video frame extraction for Apple Silicon with MLX.

Example usage:
    import viteo

    # Simple iteration
    extractor = viteo.FrameExtractor()
    extractor.open("video.mp4")
    for frame in extractor:
        # frame is an MLX array of shape (height, width, 4) with BGRA data
        process_frame(frame)

    # Or using context manager
    with viteo.open("video.mp4") as frames:
        for frame in frames:
            process_frame(frame)
"""

import mlx.core as mx
from _viteo import FrameExtractor as _FrameExtractor

__version__ = "0.3.0"
__all__ = ["FrameExtractor", "open", "extract_all"]


class FrameExtractor(_FrameExtractor):
    """
    High-performance video frame extractor using AVFoundation/VideoToolbox.

    Frames are returned as MLX arrays with shape (height, width, 4) and dtype uint8.
    The channel order is BGRA (native macOS format for best performance).
    """

    def __init__(self, path=None):
        """
        Initialize extractor and optionally open a video file.

        Args:
            path: Optional path to video file
        """
        super().__init__()
        if path:
            if not super().open(path):
                raise RuntimeError(f"Failed to open video: {path}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # C++ destructor handles cleanup automatically
        pass


def open(path):
    """
    Open a video file for frame extraction.

    Args:
        path: Path to video file

    Returns:
        FrameExtractor instance configured for iteration

    Example:
        with viteo.open("video.mp4") as frames:
            for frame in frames:
                # Process MLX array
                pass
    """
    return FrameExtractor(path)


def extract_all(path, batch_size=32):
    """
    Extract all frames from a video as a single MLX array.

    WARNING: This loads the entire video into memory. Only use for small videos.

    Args:
        path: Path to video file
        batch_size: Internal batch size for extraction (default: 32)

    Returns:
        MLX array of shape (num_frames, height, width, 4) with BGRA data
    """
    extractor = FrameExtractor(path)
    total = extractor.total_frames

    # Pre-allocate full array
    frames = mx.zeros((total, extractor.height, extractor.width, 4), dtype=mx.uint8)

    # Extract in batches directly into the array
    extracted = 0
    while extracted < total:
        batch_frames = min(batch_size, total - extracted)
        batch_view = frames[extracted:extracted + batch_frames]
        n = extractor.extract_batch_raw(batch_view, batch_frames)
        if n == 0:
            break
        extracted += n

    # Return only the frames we actually extracted
    return frames[:extracted]