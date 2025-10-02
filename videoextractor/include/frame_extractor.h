#ifndef FRAME_EXTRACTOR_H
#define FRAME_EXTRACTOR_H

#include <string>
#include <cstdint>

namespace videoextractor {

class FrameExtractor {
public:
    FrameExtractor();
    ~FrameExtractor();

    // Open video file - returns false on failure
    bool open(const std::string& path);

    // Video properties
    int width() const;
    int height() const;
    double fps() const;
    int64_t total_frames() const;

    // Extract next batch of frames directly into buffer
    // buffer: pre-allocated BGRA buffer (batch_size, height, width, 4)
    // batch_size: max frames to extract
    // Returns: number of frames actually extracted (0 when done)
    size_t extract_batch(uint8_t* buffer, size_t batch_size);

    // Reset to beginning or specific frame
    void reset(int64_t frame_index = 0);

private:
    class Impl;
    Impl* impl;
};

} // namespace videoextractor

#endif // FRAME_EXTRACTOR_H
