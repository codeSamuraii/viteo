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
    for frame in videoextractor.stream_to_iterator(extractor):
        np_frame = videoextractor.frame_to_numpy(frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(
                f"#{frame.frame_number} - {time.strftime('%H:%M:%S', time.gmtime(frame.timestamp))} ({1/((time.perf_counter() - t_start)/frame_count):.1f} fps)"
            )

        # Break early if needed
        if frame_count >= 100:
            break

    print(f"Processed {frame_count} frames using iterator\n")


if __name__ == "__main__":
    main()
