#ifndef FRAME_EXTRACTOR_H
#define FRAME_EXTRACTOR_H

#include <string>
#include <cstdint>

namespace viteo {

class FrameExtractor {
public:
    FrameExtractor();
    ~FrameExtractor();

    /**
     * @brief Opens a video file - needed before extracting
     * @param path Path to the file
     * @return A boolean indicating success or failure
     */
    bool open(const std::string& path);

    /* Video properties */
    int width() const;
    int height() const;
    double fps() const;
    int64_t total_frames() const;

    /**
     * @brief Extract next batch of frames directly into buffer
     * @param buffer Pre-allocated BGRA buffer (batch_size, height, width, 4)
     * @param batch_size Maximum frames to extract
     * @return Number of frames actually extracted (0 when done)
     */
    size_t extract_batch(uint8_t* buffer, size_t batch_size);

    /**
     * @brief Reset to beginning or specific frame
     * @param frame_index Seek to a specific index
     */
    void reset(int64_t frame_index = 0);

private:
    class Impl;
    Impl* impl;
};

} // namespace viteo

#endif // FRAME_EXTRACTOR_H
