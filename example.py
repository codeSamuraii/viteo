#!/usr/bin/env python3
"""Test script for optimized video extractor."""

import pathlib
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
    print("-" * 20, str(video_path.name), "-" * 20)

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
    print(f"* {frames_extracted} frames extracted in {elapsed:.3f}s")
    print(f"* {fps:.1f} fps / {1000*elapsed/frames_extracted:.3f}ms per frame\n")


if __name__ == "__main__":
    import sys

    if not len(sys.argv) > 1:
        print("Usage: python test_viteo.py <video_file> [<video_file> ...]")
        print("x No video file provided.")
        sys.exit(1)

    for input_path in sys.argv[1:]:
        video_path = pathlib.Path(input_path)
        if not video_path.is_file() or not video_path.exists():
            print("x Invalid path:", str(video_path))
            continue

        try:
            # test_basic_extraction(video_path)
            benchmark_extraction(video_path)
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)