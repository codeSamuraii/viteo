#ifndef FRAME_EXTRACTOR_H
#define FRAME_EXTRACTOR_H

#include <string>
#include <vector>
#include <cstdint>
#include <functional>

namespace videoextractor {

struct Frame {
    std::vector<uint8_t> data;  // BGRA data (native format, no conversion)
    int width;
    int height;
    double timestamp;
    int64_t frame_number;

    Frame() : width(0), height(0), timestamp(0.0), frame_number(0) {}
};

class FrameExtractor {
public:
    FrameExtractor();
    ~FrameExtractor();

    bool open(const std::string& path);
    void close();

    // Get video properties
    double get_duration() const;
    int get_width() const;
    int get_height() const;
    double get_fps() const;

    // High-performance streaming with batching
    // Start streaming session
    bool start_streaming(double start_time = 0.0, double end_time = 0.0);


    // Stream next batch of frames directly into a provided buffer (BGRA, uint8)
    // out_buffer: pointer to (batch, height, width, 4) buffer
    // Returns number of frames written
    size_t next_frames_batch_to_buffer(uint8_t* out_buffer, size_t max_frames, int width, int height, size_t stride);

    // Get next batch of frames (up to max_frames)
    // Returns actual number of frames retrieved
    size_t next_frames_batch(std::vector<Frame>& frames, size_t max_frames = 32);

    // Check if streaming session is active
    bool is_streaming() const;

private:
    class Impl;
    Impl* pImpl;
};

} // namespace videoextractor

#endif // FRAME_EXTRACTOR_H
