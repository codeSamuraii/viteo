"""Hardware-accelerated video frame extraction for Apple Silicon."""

from .build._videoextractor import FrameExtractor, Frame
import numpy as np
from typing import Iterator, Callable, Optional
from collections import deque

__version__ = "0.1.0"
__all__ = ["FrameExtractor", "Frame", "frame_to_numpy", "stream_to_iterator", "stream_to_queue"]


def frame_to_numpy(frame: Frame) -> np.ndarray:
    """Convert a Frame object to a numpy array.

    Args:
        frame: Frame object from the extractor

    Returns:
        numpy array with shape (height, width, 3) and dtype uint8
    """
    data = frame.data
    height, width = frame.height, frame.width

    # Convert bytes to numpy array
    arr = np.frombuffer(data, dtype=np.uint8)

    # Reshape to (height, width, 3)
    return arr.reshape((height, width, 3))


def stream_to_iterator(extractor: FrameExtractor, start_time: float = 0.0, end_time: float = 0.0) -> Iterator[Frame]:
    """Convert streaming API to Python iterator.

    Args:
        extractor: FrameExtractor instance with an open video
        start_time: Start timestamp in seconds (default: 0.0 for beginning)
        end_time: End timestamp in seconds (default: 0.0 for end of video)

    Yields:
        Frame objects as they are decoded

    Example:
        >>> extractor = FrameExtractor()
        >>> extractor.open("video.mp4")
        >>> for frame in stream_to_iterator(extractor):
        ...     arr = frame_to_numpy(frame)
        ...     # Process frame...
    """
    frames = deque()

    def callback(frame: Frame) -> bool:
        frames.append(frame)
        return True

    # Start streaming in callback
    if end_time > 0:
        extractor.stream_frames_range(start_time, end_time, callback)
    elif start_time > 0:
        extractor.stream_frames_from(start_time, callback)
    else:
        extractor.stream_frames(callback)

    # Yield all collected frames
    while frames:
        yield frames.popleft()


def stream_to_queue(extractor: FrameExtractor, maxsize: int = 0, start_time: float = 0.0, end_time: float = 0.0):
    """Stream frames to a queue for producer-consumer pattern.

    Args:
        extractor: FrameExtractor instance with an open video
        maxsize: Maximum queue size (0 for unlimited)
        start_time: Start timestamp in seconds (default: 0.0 for beginning)
        end_time: End timestamp in seconds (default: 0.0 for end of video)

    Returns:
        Queue object that will be filled with Frame objects

    Example:
        >>> import threading
        >>> from queue import Queue
        >>>
        >>> extractor = FrameExtractor()
        >>> extractor.open("video.mp4")
        >>> q = stream_to_queue(extractor, maxsize=10)
        >>>
        >>> # Consumer thread
        >>> while True:
        ...     frame = q.get()
        ...     if frame is None:  # Sentinel value for end
        ...         break
        ...     # Process frame...
    """
    from queue import Queue
    import threading

    q = Queue(maxsize=maxsize)

    def stream_thread():
        try:
            def callback(frame: Frame) -> bool:
                q.put(frame)
                return True

            if end_time > 0:
                extractor.stream_frames_range(start_time, end_time, callback)
            elif start_time > 0:
                extractor.stream_frames_from(start_time, callback)
            else:
                extractor.stream_frames(callback)
        finally:
            # Signal end of stream
            q.put(None)

    thread = threading.Thread(target=stream_thread, daemon=True)
    thread.start()

    return q
