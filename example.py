#!/usr/bin/env python3
"""Test script for optimized video extractor."""

import time
import mlx.core as mx
import viteo

def test_basic_extraction(video_path):
    """Test basic frame extraction."""
    print(f"\nTesting: {video_path}")
    print("-" * 50)

    # Open video
    extractor = viteo.FrameExtractor(video_path)
    print(f"Video properties:")
    print(f"  Resolution: {extractor.width}x{extractor.height}")
    print(f"  FPS: {extractor.fps:.2f}")
    print(f"  Total frames: {extractor.total_frames}")

    # Test iteration
    print("\nExtracting first 100 frames...")
    start = time.time()
    frames_extracted = 0

    for frame in extractor:
        frames_extracted += 1
        if frames_extracted == 1:
            print(f"  First frame shape: {frame.shape}, dtype: {frame.dtype}")
        if frames_extracted >= 100:
            break

    elapsed = time.time() - start
    fps = frames_extracted / elapsed
    print(f"  Extracted {frames_extracted} frames in {elapsed:.3f}s")
    print(f"  Extraction speed: {fps:.1f} fps")

    # Test reset
    print("\nTesting reset...")
    extractor.reset()
    frame_after_reset = next(iter(extractor))
    print(f"  First frame after reset: shape={frame_after_reset.shape}")

    # Test context manager
    print("\nTesting context manager...")
    with viteo.open(video_path) as frames:
        first_frame = next(iter(frames))
        print(f"  Got frame with shape: {first_frame.shape}")

    print("\n✓ All tests passed!")


def benchmark_extraction(video_path, num_frames=500):
    """Benchmark frame extraction speed."""
    print(f"\nBenchmarking extraction of {num_frames} frames...")
    print("-" * 50)

    extractor = viteo.FrameExtractor(video_path)

    # Warm up
    for i, frame in enumerate(extractor):
        if i >= 10:
            break

    # Reset and benchmark
    extractor.reset()
    start = time.time()
    frames_extracted = 0

    for frame in extractor:
        frames_extracted += 1
        # Ensure frame is evaluated (MLX is lazy)
        mx.eval(frame)
        if frames_extracted >= num_frames:
            break

    elapsed = time.time() - start
    fps = frames_extracted / elapsed

    print(f"Results:")
    print(f"  Frames extracted: {frames_extracted}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Speed: {fps:.1f} fps")
    print(f"  Per frame: {1000*elapsed/frames_extracted:.2f}ms")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Try to find a sample video
        import os
        possible_paths = [
            "sample.mp4",
            "test.mp4",
            "video.mp4",
            "../sample.mp4",
            "../test.mp4",
            "../video.mp4"
        ]
        video_path = None
        for path in possible_paths:
            if os.path.exists(path):
                video_path = path
                break

        if not video_path:
            print("Usage: python test_viteo.py <video_file>")
            print("No video file provided and no sample video found.")
            sys.exit(1)

    try:
        test_basic_extraction(video_path)
        benchmark_extraction(video_path)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)