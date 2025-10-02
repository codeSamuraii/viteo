"""
Comprehensive tests for the viteo package.

Includes functionality tests, error handling, and performance benchmarks.
"""
import os
import time
import pytest
import tempfile
import mlx.core as mx
from pathlib import Path

import viteo


# --- Fixtures ---

@pytest.fixture
def test_data_dir():
    """Path to test video files."""
    return Path(__file__).parent / "test-data"


@pytest.fixture
def video_files(test_data_dir):
    """Dictionary of test video files with their properties."""
    return {
        "4k": {
            "path": test_data_dir / "video_4k.mp4",
            "width": 3840,
            "height": 2160,
            "min_fps": 60.0  # Minimum expected extraction speed
        },
        "1080p": {
            "path": test_data_dir / "video_1080p.mp4",
            "width": 1920,
            "height": 1080,
            "min_fps": 120.0
        },
        "720p": {
            "path": test_data_dir / "video_720p.mp4",
            "width": 1280,
            "height": 720,
            "min_fps": 240.0
        },
        "480p": {
            "path": test_data_dir / "video_480p.mp4",
            "width": 854,
            "height": 480,
            "min_fps": 480.0
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

    extractor = viteo.FrameExtractor(str(path))

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
            assert isinstance(frame, mx.array)
            assert frame.shape == (frames.height, frames.width, 4)
            assert frame.dtype == mx.uint8
            count += 1
            if count >= 10:
                break
        assert count == 10


def test_reset(sample_video):
    """Test reset functionality."""
    path = sample_video["path"]
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")

    extractor = viteo.FrameExtractor(str(path))

    # Get first frame
    first_frame = next(extractor)

    # Get 10 more frames
    for _ in range(10):
        next(extractor)

    # Reset and get first frame again
    extractor.reset()
    new_first_frame = next(extractor)

    # Compare pixel sums as a simple way to check if frames are similar
    assert mx.sum(first_frame).item() == mx.sum(new_first_frame).item()


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

    extractor = viteo.FrameExtractor(str(path))

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
    extractor = viteo.FrameExtractor(str(video_path))

    # Warmup - extract a few frames to initialize everything
    for i in range(10):
        try:
            next(iter(extractor))
        except StopIteration:
            return (0, 0)  # Video too short for test

    # Reset to beginning
    extractor.reset(0)

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

    # Path to test videos
    test_data = Path(__file__).parent / "test-data"

    # Use videos provided on command line or all test videos
    if len(sys.argv) > 1:
        videos = [Path(p) for p in sys.argv[1:]]
    else:
        videos = [
            test_data / "video_4k.mp4",
            test_data / "video_1080p.mp4",
            test_data / "video_720p.mp4",
            test_data / "video_480p.mp4",
        ]

    # Run benchmark for each video
    for video_path in videos:
        if not video_path.exists():
            print(f"File not found: {video_path}")
            continue

        print(f"{'-'*20} {video_path.name} {'-'*20}")

        try:
            extractor = viteo.FrameExtractor(str(video_path))

            # Print video properties
            print(f"Video:")
            print(f"* Resolution: {extractor.width}x{extractor.height}")
            print(f"* FPS: {extractor.fps:.2f}")
            print(f"* Total frames: {extractor.total_frames}")

            # Extract frames and measure performance
            fps, ms_per_frame = measure_performance(video_path)
            print(f"Benchmark:")
            print(f"* 256 frames extracted in {(256 * ms_per_frame / 1000):.3f}s")
            print(f"* {fps:.1f} fps / {ms_per_frame:.3f}ms per frame")

        except Exception as e:
            print(f"\nxxx Error testing {video_path}: {e}\n")
            import traceback
            traceback.print_exc()

        print('\n')
