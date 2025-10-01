
import mlx.core as mx
from .build._videoextractor import FrameExtractor

__version__ = "0.2.0"
__all__ = ["FrameExtractor", "stream_frames_mlx"]


def stream_frames_mlx(
    extractor: FrameExtractor,
    start_time: float = 0.0,
    end_time: float = 0.0,
    internal_batch: int = 32
):
    """
    Generator that yields MLX frames (BGRA, uint8) one at a time for maximum performance.
    Args:
        extractor: FrameExtractor instance with an open video
        start_time: Start timestamp in seconds (default: 0.0)
        end_time: End timestamp in seconds (default: 0.0 for end of video)
        internal_batch: Internal batch size for efficiency (default: 32)
    Yields:
        MLX array of shape (height, width, 4), dtype=uint8 for each frame
    """
    if not extractor.start_streaming(start_time, end_time):
        raise RuntimeError("Failed to start streaming")
    width = extractor.width
    height = extractor.height
    buf = mx.zeros((internal_batch, height, width, 4), dtype=mx.uint8)
    while extractor.is_streaming():
        frames_written = extractor.next_frames_batch_to_buffer(buf, internal_batch)
        if frames_written == 0:
            break
        for i in range(frames_written):
            yield buf[i]