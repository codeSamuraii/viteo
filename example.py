import time

import videoextractor


def main():
    print("=" * 60)
    print("Stream using Python iterator")
    print("=" * 60)

    video_path = "videos/video_4k.mp4"
    extractor = videoextractor.FrameExtractor()
    if not extractor.open(video_path):
        print(f"Failed to open video: {video_path}")
        return

    frame_count, t_start = 0, time.perf_counter()

    for frame in videoextractor.stream_frames_mlx(extractor=extractor):
        frame_count += 1
        if frame_count % 100 == 0:
            print(
                f"#{frame_count} - ({1/((time.perf_counter() - t_start)/frame_count):.1f} fps) - {type(frame)} {frame.shape} {frame.dtype}"
            )

        # Break early if needed
        if frame_count >= 10000:
            break

    print(f"Processed {frame_count} frames using iterator\n")


if __name__ == "__main__":
    main()
