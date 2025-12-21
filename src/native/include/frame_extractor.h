#ifndef FRAME_EXTRACTOR_H
#define FRAME_EXTRACTOR_H

#include <string>
#include <cstdint>
#include <vector>
#include <memory>

namespace viteo {

/// High-performance video frame extractor for Apple Silicon
class FrameExtractor {
public:
    FrameExtractor();
    ~FrameExtractor();

    /// Open video file for extraction
    bool open(const std::string& path);

    /// Get next frame buffer (nullptr when done)
    std::shared_ptr<std::vector<uint8_t>> next_frame();

    /// Reset to beginning or specific frame index
    void reset(int64_t frame_index = 0);

    /// Video width in pixels
    int width() const;

    /// Video height in pixels
    int height() const;

    /// Video frames per second
    double fps() const;

    /// Estimated total number of frames
    int64_t total_frames() const;

private:
    class Impl;
    Impl* impl;
};

} // namespace viteo

#endif // FRAME_EXTRACTOR_H
