import pathlib
import time
import mlx.core as mx
import viteo
import cv2


def benchmark_extraction(video_path):
    """Test basic frame extraction."""

    with viteo.open(video_path) as video:
        print(f"---- {video_path.name} ----")
        print(f"> Resolution: {video.width}x{video.height}")
        print(f"> FPS: {video.fps:.2f}")
        print(f"> Total frames: {video.total_frames}")

        print("* Running benchmark...", end='\r')
        frames_extracted = 0
        num_frames = min(256, video.total_frames)
        start = time.time()

        for frame in video:
            frames_extracted += 1
            mx.eval(frame)
            if frames_extracted >= num_frames:
                break

        elapsed = time.time() - start
        fps = frames_extracted / elapsed

        print(f"> {frames_extracted} frames extracted in {elapsed:.3f}s")
        print(f"> {fps:.1f} fps / {1000*elapsed/frames_extracted:.3f}ms per frame\n")


def benchmark_extraction_opencv(video_path):
    """Benchmark frame extraction speed using OpenCV."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"---- {video_path.name} (OpenCV) ----")
    print(f"> Resolution: {width}x{height}")
    print(f"> FPS: {fps:.2f}")
    print(f"> Total frames: {total_frames}")

    print("* Running benchmark...", end='\r')
    frames_extracted = 0
    num_frames = min(256, total_frames)
    start = time.time()

    while frames_extracted < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames_extracted += 1

    elapsed = time.time() - start
    fps = frames_extracted / elapsed
    cap.release()

    print(f"> {frames_extracted} frames extracted in {elapsed:.3f}s")
    print(f"> {fps:.1f} fps / {1000*elapsed/frames_extracted:.3f}ms per frame\n")


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
            benchmark_extraction(video_path)
            benchmark_extraction_opencv(video_path)
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)