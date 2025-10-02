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
from _viteo import FrameExtractor as _FrameExtractor, FrameIterator as _FrameIterator
from typing import Optional, Iterator

__version__ = "0.1.0c"
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
        self._iterator = None
        self._batch_size = 32

        if path:
            if not super().open(str(path)):
                raise RuntimeError(f"Failed to open video: {path}")

    def _create_iterator(self):
        """Create a new FrameIterator with a fresh buffer."""
        buffer = mx.zeros((self._batch_size, self.height, self.width, 4), mx.uint8)
        return _FrameIterator(self, buffer, self._batch_size)

    def reset(self, frame_index: int = 0):
        """
        Reset to beginning or specific frame.

        Args:
            frame_index: Frame index to seek to (default: 0)
        """
        super().reset(frame_index)
        # Reset the iterator state
        self._iterator = None

    def __iter__(self):
        """Return self as iterator."""
        # Create iterator on first call or after reset
        if self._iterator is None:
            self._iterator = self._create_iterator()
        return self

    def __next__(self):
        """Get next frame."""
        if self._iterator is None:
            self._iterator = self._create_iterator()
        return next(self._iterator)

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
