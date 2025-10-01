#ifndef FRAME_EXTRACTOR_H
#define FRAME_EXTRACTOR_H

#include <string>
#include <vector>
#include <cstdint>
#include <functional>

namespace videoextractor {

struct Frame {
    std::vector<uint8_t> data;  // RGB data
    int width;
    int height;
    double timestamp;  // in seconds
    int64_t frame_number;  // Sequential frame number

    Frame() : width(0), height(0), timestamp(0.0), frame_number(0) {}
};

class FrameExtractor {
public:
    FrameExtractor();
    ~FrameExtractor();

    // Open a video file
    bool open(const std::string& path);

    // Close the current video
    void close();

    // Extract a single frame at the given timestamp (in seconds)
    Frame extract_frame(double timestamp);

    // Extract frames at specified timestamps
    std::vector<Frame> extract_frames(const std::vector<double>& timestamps);

    // Extract frames at regular intervals
    std::vector<Frame> extract_frames_interval(double start, double end, double interval);

    // Get video properties
    double get_duration() const;
    int get_width() const;
    int get_height() const;
    double get_fps() const;

    // Streaming API - read frames sequentially as fast as possible
    // Callback is called for each frame as it's decoded
    // Returns false if streaming should stop, true to continue
    using FrameCallback = std::function<bool(const Frame&)>;

    // Start streaming frames from the beginning
    void stream_frames(FrameCallback callback);

    // Start streaming frames from a specific timestamp
    void stream_frames_from(double start_time, FrameCallback callback);

    // Start streaming frames between start and end timestamps
    void stream_frames_range(double start_time, double end_time, FrameCallback callback);

private:
    class Impl;
    Impl* pImpl;
};

} // namespace videoextractor

#endif // FRAME_EXTRACTOR_H
