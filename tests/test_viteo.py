"""
Comprehensive tests for the viteo package.

Includes functionality tests, error handling, and performance benchmarks.
"""
import os
import time
import pytest
import tempfile
from pathlib import Path

import viteo


# --- Fixtures ---

@pytest.fixture
def test_data_dir():
    """Path to test video files."""
    return Path(__file__).parent / "samples"


@pytest.fixture
def video_files(test_data_dir):
    """Dictionary of test video files with their properties."""
    return {
        "720p": {
            "path": test_data_dir / "720p.mp4",
            "width": 1280,
            "height": 720,
            "min_fps": 200.0
        },
        "1080p": {
            "path": test_data_dir / "1080p.mp4",
            "width": 1920,
            "height": 1080,
            "min_fps": 120.0
        },
        "4k": {
            "path": test_data_dir / "4k.mp4",
            "width": 3840,
            "height": 2160,
            "min_fps": 60.0
        },
        "8k": {
            "path": test_data_dir / "8k.mp4",
            "width": 7680,
            "height": 4320,
            "min_fps": 30.0
        }
    }


@pytest.fixture
def sample_video(video_files):
    """A standard test video file (720p for faster tests)."""
    return video_files["720p"]


# --- Basic Functionality Tests ---

def test_open_video(sample_video):
    """Test opening a video file."""
    path = sample_video["path"]
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")

    extractor = viteo.FrameExtractor()
    assert extractor.open(str(path)) == True

    # Check that properties are set correctly
    assert extractor.width > 0
    assert extractor.height > 0
    assert extractor.fps > 0
    assert extractor.total_frames > 0


def test_constructor_with_path(sample_video):
    """Test constructor with path parameter."""
    path = sample_video["path"]
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")

    extractor = viteo.open(path)

    # Check that properties are set correctly
    assert extractor.width > 0
    assert extractor.height > 0
    assert extractor.fps > 0
    assert extractor.total_frames > 0


def test_context_manager(sample_video):
    """Test using the context manager."""
    path = sample_video["path"]
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")

    with viteo.open(str(path)) as frames:
        assert frames.width > 0
        assert frames.height > 0
        assert frames.fps > 0
        assert frames.total_frames > 0


def test_iterator(sample_video):
    """Test iterating through frames."""
    path = sample_video["path"]
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")

    with viteo.open(str(path)) as frames:
        # Get first 10 frames
        count = 0
        for frame in frames:
            mv = memoryview(frame)
            assert mv.shape == (frames.height, frames.width, frames.channels)
            assert mv.format == 'B'
            count += 1
            if count >= 10:
                break
        assert count == 10


def test_run_to_end(sample_video):
    """Test running through all frames to the end."""
    path = sample_video["path"]
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")

    with viteo.open(path) as video:
        frame_count = 0
        for frame in video:
            frame_count += 1

    assert abs(frame_count - video.total_frames) <= 1


def test_last_frame_is_none(sample_video):
    """Test that after all frames are read, the next frame is None and the frame count matches total_frames."""
    path = sample_video["path"]
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")

    i = 0
    with viteo.open(path) as video:
        while True:
            frame = video.next_frame()
            if frame is None:
                break

            i += 1

        assert frame is None
        assert abs(i - video.total_frames) <= 1


def test_reset(sample_video):
    """Test reset functionality."""
    path = sample_video["path"]
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")

    extractor = viteo.open(path)

    # Get first frame
    first_frame = memoryview(next(extractor))

    # Get 10 more frames
    for _ in range(10):
        next(extractor)

    # Reset and get first frame again
    extractor.reset()
    new_first_frame = memoryview(next(extractor))

    # Compare pixel sums as a simple way to check if frames are similar
    assert sum(first_frame.cast('B')) == sum(new_first_frame.cast('B'))


def test_properties(video_files):
    """Test video properties match expected resolutions."""
    for res_name, video_info in video_files.items():
        path = video_info["path"]
        if not path.exists():
            pytest.skip(f"Test video not found: {path}")

        expected_width = video_info["width"]
        expected_height = video_info["height"]

        with viteo.open(str(path)) as frames:
            # Allow small variations in resolution (±10 pixels)
            assert abs(frames.width - expected_width) <= 10, f"Width mismatch for {res_name}"
            assert abs(frames.height - expected_height) <= 10, f"Height mismatch for {res_name}"


# --- Error Handling Tests ---

def test_nonexistent_file():
    """Test behavior with nonexistent file."""
    with pytest.raises(RuntimeError):
        viteo.FrameExtractor("/nonexistent/path/to/video.mp4")


def test_invalid_file():
    """Test behavior with invalid file."""
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(suffix=".mp4", mode="w") as f:
        f.write("This is not a video file")
        f.flush()

        with pytest.raises(RuntimeError):
            viteo.FrameExtractor(f.name)


def test_reset_out_of_bounds(sample_video):
    """Test reset with out-of-bounds frame index."""
    path = sample_video["path"]
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")

    extractor = viteo.open(path)

    # Reset to a frame index way beyond the end of the video
    extractor.reset(1000000)

    # Trying to get a frame should not crash but might return no frames
    iterator = iter(extractor)
    try:
        frame = next(iterator)
        # If we got a frame, that's fine too - the implementation might clamp to valid range
    except StopIteration:
        # This is an expected outcome
        pass


# --- Performance Tests ---

def measure_performance(video_path, num_frames=200):
    """
    Measure the performance of frame extraction.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract

    Returns:
        tuple: (frames_per_second, ms_per_frame)
    """
    extractor = viteo.FrameExtractor(video_path)

    # Time the extraction of frames
    start_time = time.time()

    frame_count = 0
    for frame in extractor:
        frame_count += 1
        if frame_count >= num_frames:
            break

    end_time = time.time()

    if frame_count < num_frames:
        pytest.skip(f"Video has fewer than {num_frames} frames")

    duration = end_time - start_time
    fps = frame_count / duration
    ms_per_frame = (duration / frame_count) * 1000

    return (fps, ms_per_frame)


@pytest.mark.slow
def test_performance(video_files):
    """Test performance for all test videos."""
    for res_name, video_info in video_files.items():
        path = video_info["path"]
        if not path.exists():
            pytest.skip(f"Test video not found: {path}")

        fps, ms_per_frame = measure_performance(path)
        min_fps = video_info["min_fps"]

        print(f"\nResults for {res_name}: {fps:.1f} fps / {ms_per_frame:.3f}ms per frame")
        print(f"Required: {min_fps:.1f} fps minimum")

        # Only fail if performance is significantly below threshold (80% of expected)
        assert fps >= min_fps * 0.8, f"Performance below threshold for {res_name}: got {fps:.1f} fps, expected at least {min_fps:.1f} fps"


# --- Utility Code ---

if __name__ == "__main__":
    """Run benchmarks when executed directly."""
    import os
    import sys

    if len(sys.argv) > 1:
        videos = [Path(p) for p in sys.argv[1:]]
    else:
        samples_dir = Path(__file__).parent / "samples"
        videos = list(samples_dir.rglob("*.mp4", case_sensitive=False))

    # Run benchmark for each video
    for video_path in videos:
        if not video_path.is_file():
            print(f"x Not found: {video_path}")
            continue

        extractor = viteo.open(video_path)
        num_frames = min(256, extractor.total_frames)
        fps, ms_per_frame = measure_performance(video_path, num_frames)
        print(f"* {video_path.name}: {fps:.2f} fps - {ms_per_frame:.2f}ms")
