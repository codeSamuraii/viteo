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
import pathlib
import mlx.core as mx
from _viteo import FrameExtractor as _FrameExtractor
from typing import Optional, Iterator

__version__ = "0.1.0"
__all__ = ["FrameExtractor", "open"]


class FrameExtractor(_FrameExtractor):
    """
    Hardware-accelerated video frame extractor for Apple Silicon.

    Frames are returned as MLX arrays with BGRA channels and uint8 data type.
    Frames are buffered internally and passed in batches to reduce overhead from C++ bindings.
    """

    def __init__(self, path: Optional[str | pathlib.Path] = None):
        """
        Initialize extractor and optionally open a video file.

        Args:
            path: Optional path to video file
        """
        super().__init__()
        if path:
            if not super().open(str(path)):
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
