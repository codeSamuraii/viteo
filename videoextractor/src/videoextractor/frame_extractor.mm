
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>
#import <VideoToolbox/VideoToolbox.h>
#include "frame_extractor.h"
#include <memory>

namespace videoextractor {

class FrameExtractor::Impl {
public:
    AVAsset* asset;
    AVAssetReader* reader;
    AVAssetReaderTrackOutput* output;
    AVAssetTrack* videoTrack;
    int64_t currentFrameNumber;

    Impl() : asset(nil), reader(nil), output(nil), videoTrack(nil), currentFrameNumber(0) {}
    ~Impl() { close(); }

    void close() {
        if (reader) {
            [reader cancelReading];
            reader = nil;
        }
        output = nil;
        videoTrack = nil;
        asset = nil;
        currentFrameNumber = 0;
    }

    bool open(const std::string& path) {
        close();
        NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:nsPath];
        asset = [AVAsset assetWithURL:url];
        if (!asset) return false;
        NSArray* tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
        if (tracks.count == 0) return false;
        videoTrack = tracks[0];
        return videoTrack != nil;
    }

    int getWidth() const {
        if (!videoTrack) return 0;
        CGSize size = [videoTrack naturalSize];
        return static_cast<int>(size.width);
    }
    int getHeight() const {
        if (!videoTrack) return 0;
        CGSize size = [videoTrack naturalSize];
        return static_cast<int>(size.height);
    }
    double getFPS() const {
        if (!videoTrack) return 0.0;
        return static_cast<double>([videoTrack nominalFrameRate]);
    }

    bool start_streaming(double start_time, double end_time) {
        if (!asset || !videoTrack) return false;
        if (reader) {
            [reader cancelReading];
            reader = nil;
            output = nil;
        }
        @autoreleasepool {
            NSError* error = nil;
            reader = [[AVAssetReader alloc] initWithAsset:asset error:&error];
            if (error || !reader) return false;
            NSDictionary* outputSettings = @{
                (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
                (id)kCVPixelBufferMetalCompatibilityKey: @YES,
            };
            output = [[AVAssetReaderTrackOutput alloc] initWithTrack:videoTrack outputSettings:outputSettings];
            output.alwaysCopiesSampleData = NO;
            if (![reader canAddOutput:output]) {
                reader = nil;
                output = nil;
                return false;
            }
            [reader addOutput:output];
            if (start_time > 0 || end_time > 0) {
                CMTime start = CMTimeMakeWithSeconds(start_time, 600);
                CMTime duration;
                if (end_time > 0) {
                    duration = CMTimeMakeWithSeconds(end_time - start_time, 600);
                } else {
                    duration = CMTimeSubtract([asset duration], start);
                }
                reader.timeRange = CMTimeRangeMake(start, duration);
            }
            if (![reader startReading]) {
                reader = nil;
                output = nil;
                return false;
            }
            currentFrameNumber = 0;
            return true;
        }
    }

    // Write up to max_frames directly into out_buffer (BGRA, uint8), returns frames written
    size_t next_frames_batch_to_buffer(uint8_t* out_buffer, size_t max_frames, int width, int height, size_t stride) {
        if (!reader || !output) return 0;
        if (reader.status != AVAssetReaderStatusReading) {
            [reader cancelReading];
            reader = nil;
            output = nil;
            return 0;
        }
        size_t frames_written = 0;
        @autoreleasepool {
            for (size_t i = 0; i < max_frames; i++) {
                CMSampleBufferRef sampleBuffer = [output copyNextSampleBuffer];
                if (!sampleBuffer) {
                    [reader cancelReading];
                    reader = nil;
                    output = nil;
                    break;
                }
                CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
                if (!imageBuffer) {
                    CFRelease(sampleBuffer);
                    break;
                }
                CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
                size_t src_width = CVPixelBufferGetWidth(imageBuffer);
                size_t src_height = CVPixelBufferGetHeight(imageBuffer);
                size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
                uint8_t* baseAddress = (uint8_t*)CVPixelBufferGetBaseAddress(imageBuffer);
                // Write to out_buffer at offset
                uint8_t* dst = out_buffer + frames_written * stride;
                if (bytesPerRow == width * 4 && src_width == (size_t)width && src_height == (size_t)height) {
                    memcpy(dst, baseAddress, width * height * 4);
                } else {
                    // Copy row by row
                    for (size_t y = 0; y < height && y < src_height; y++) {
                        memcpy(dst + y * width * 4, baseAddress + y * bytesPerRow, width * 4);
                    }
                }
                CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
                CFRelease(sampleBuffer);
                frames_written++;
            }
        }
        return frames_written;
    }

    bool is_streaming() const {
        return reader != nil && reader.status == AVAssetReaderStatusReading;
    }
};

// FrameExtractor implementation
FrameExtractor::FrameExtractor() : pImpl(new Impl()) {}
FrameExtractor::~FrameExtractor() { delete pImpl; }
bool FrameExtractor::open(const std::string& path) { return pImpl->open(path); }
void FrameExtractor::close() { pImpl->close(); }
int FrameExtractor::get_width() const { return pImpl->getWidth(); }
int FrameExtractor::get_height() const { return pImpl->getHeight(); }
double FrameExtractor::get_fps() const { return pImpl->getFPS(); }
bool FrameExtractor::start_streaming(double start_time, double end_time) { return pImpl->start_streaming(start_time, end_time); }
size_t FrameExtractor::next_frames_batch_to_buffer(uint8_t* out_buffer, size_t max_frames, int width, int height, size_t stride) {
    return pImpl->next_frames_batch_to_buffer(out_buffer, max_frames, width, height, stride);
}
bool FrameExtractor::is_streaming() const { return pImpl->is_streaming(); }

} // namespace videoextractor
