"""
Hardware-accelerated video frame extraction for Apple Silicon with MLX.

Example usage:
    import viteo

    # Context manager style
    with viteo.open("video.mp4") as frames:
        for frame in frames:
            process_frame(frame)

    # Direct iteration
    extractor = viteo.FrameExtractor()
    extractor.open("video.mp4")
    for frame in extractor:
        process_frame(frame)
"""
import pathlib
from _viteo import FrameExtractor as _FrameExtractor
from typing import Optional

__version__ = "0.1.1"
__all__ = ["FrameExtractor", "open"]


class FrameExtractor(_FrameExtractor):
    """Hardware-accelerated video frame extractor for Apple Silicon."""

    def __init__(self, path: Optional[str | pathlib.Path] = None):
        """Initialize extractor and optionally open a video file."""
        super().__init__()
        if path:
            if not super().open(str(path)):
                raise RuntimeError(f"Failed to open video: {path}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def open(path: str | pathlib.Path) -> FrameExtractor:
    """
    Open a video file for frame extraction.

    Args:
        path: Path to video file

    Returns:
        FrameExtractor configured for iteration

    Example:
        with viteo.open("video.mp4") as frames:
            for frame in frames:
                process_frame(frame)
    """
    return FrameExtractor(path)
