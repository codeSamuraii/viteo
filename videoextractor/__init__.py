"""Hardware-accelerated video frame extraction for Apple Silicon."""

from .build._videoextractor import FrameExtractor, Frame
import numpy as np
from typing import Iterator, Optional

__version__ = "0.1.0"
__all__ = ["FrameExtractor", "Frame", "frame_to_numpy", "frame_to_mlx", "stream_frames"]


def frame_to_numpy(frame: Frame, format: str = "bgra") -> np.ndarray:
    """Convert a Frame object to a numpy array.

    Args:
        frame: Frame object from the extractor
        format: Output format - "bgra" (default, fastest), "rgb", or "rgba"

    Returns:
        numpy array with shape (height, width, channels) and dtype uint8
    """
    data = frame.data
    height, width = frame.height, frame.width

    arr = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))

    if format == "bgra":
        return arr
    elif format == "rgb":
        # BGRA -> RGB (slower due to conversion)
        return arr[:, :, [2, 1, 0]]
    elif format == "rgba":
        # BGRA -> RGBA
        return arr[:, :, [2, 1, 0, 3]]
    else:
        raise ValueError(f"Unknown format: {format}")


def frame_to_mlx(frame: Frame):
    """Convert a Frame object to an MLX array (zero-copy when possible).

    Args:
        frame: Frame object from the extractor

    Returns:
        mlx.core.array with shape (height, width, 4) and dtype uint8 (BGRA format)
    """
    try:
        import mlx.core as mx
    except ImportError:
        raise ImportError("MLX is not installed. Install it with: pip install mlx")

    # Convert to numpy first (minimal copy)
    arr = frame_to_numpy(frame, format="bgra")

    # MLX can use the numpy array directly on Apple Silicon
    return mx.array(arr)


def stream_frames(
    extractor: FrameExtractor,
    start_time: float = 0.0,
    end_time: float = 0.0,
    batch_size: int = 32
) -> Iterator[Frame]:
    """Stream frames in batches for maximum performance.

    This generator uses batch processing to minimize Python overhead.
    Frames are decoded in BGRA format (native) for maximum speed.

    Args:
        extractor: FrameExtractor instance with an open video
        start_time: Start timestamp in seconds (default: 0.0 for beginning)
        end_time: End timestamp in seconds (default: 0.0 for end of video)
        batch_size: Number of frames to decode per batch (default: 32)

    Yields:
        Frame objects as they are decoded from the video

    Example:
        >>> extractor = FrameExtractor()
        >>> extractor.open("video.mp4")
        >>> for frame in stream_frames(extractor, batch_size=64):
        ...     arr = frame_to_mlx(frame)  # Fast BGRA to MLX
        ...     # Process frame with MLX...
        ...     if some_condition:
        ...         break
    """
    if not extractor.start_streaming(start_time, end_time):
        return

    while extractor.is_streaming():
        frames = extractor.next_frames_batch(batch_size)

        if not frames:
            break

        for frame in frames:
            yield frame