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

    def __init__(self, path: Optional[str | pathlib.Path] = None, batch_size: int = 8):
        """
        Initialize extractor and optionally open a video file.

        Args:
            path: Optional path to video file
            batch_size: Number of frames to buffer internally (default: 8)
        """
        super().__init__(batch_size)
        self.batch_size = batch_size
        if path:
            if not super().open(str(path)):
                raise RuntimeError(f"Failed to open video: {path}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def open(path: str | pathlib.Path, batch_size: int = 8) -> FrameExtractor:
    """
    Open a video file for frame extraction.

    Args:
        path: Path to video file
        batch_size: Number of frames to buffer internally (default: 8)

    Returns:
        FrameExtractor configured for iteration

    Example:
        with viteo.open("video.mp4", batch_size=16) as frames:
            for frame in frames:
                process_frame(frame)
    """
    return FrameExtractor(path, batch_size=batch_size)
