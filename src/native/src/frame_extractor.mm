#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>
#import <VideoToolbox/VideoToolbox.h>
#include "frame_extractor.h"

namespace viteo {

/// Internal implementation with AVFoundation
class FrameExtractor::Impl {
public:
    AVAsset* asset = nil;
    AVAssetReader* reader = nil;
    AVAssetReaderTrackOutput* output = nil;
    AVAssetTrack* videoTrack = nil;

    int cachedWidth = 0;
    int cachedHeight = 0;
    double cachedFPS = 0.0;
    int64_t cachedTotalFrames = 0;
    int64_t currentFrame = 0;

    // Internal batch buffer for performance
    static constexpr size_t BATCH_SIZE = 16;
    std::vector<uint8_t> batch_buffer;
    size_t batch_count = 0;
    size_t batch_index = 0;

    bool isOpen = false;

    Impl() {}

    ~Impl() {
        close();
        // ARC handles cleanup automatically
    }

    void close() {
        @autoreleasepool {
            if (reader) {
                [reader cancelReading];
                reader = nil;
            }
            output = nil;
            videoTrack = nil;
            asset = nil;
            isOpen = false;
            currentFrame = 0;
        }
    }

    bool open(const std::string& path) {
        close();

        @autoreleasepool {
            NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
            NSURL* url = [NSURL fileURLWithPath:nsPath];

            asset = [AVAsset assetWithURL:url];
            if (!asset) return false;

            #pragma clang diagnostic push
            #pragma clang diagnostic ignored "-Wdeprecated-declarations"
            NSArray* tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
            #pragma clang diagnostic pop

            if (tracks.count == 0) return false;

            videoTrack = tracks[0];

            CGSize size = [videoTrack naturalSize];
            cachedWidth = static_cast<int>(size.width);
            cachedHeight = static_cast<int>(size.height);
            cachedFPS = [videoTrack nominalFrameRate];

            CMTime duration = [asset duration];
            cachedTotalFrames = static_cast<int64_t>(
                CMTimeGetSeconds(duration) * cachedFPS
            );

            // Allocate batch buffer
            size_t frame_size = cachedWidth * cachedHeight * 4;
            batch_buffer.resize(BATCH_SIZE * frame_size);

            isOpen = true;
            return setupReader(0);
        }
    }

    bool setupReader(int64_t startFrame) {
        @autoreleasepool {
            if (reader) {
                [reader cancelReading];
                reader = nil;
                output = nil;
            }

            NSError* error = nil;
            reader = [[AVAssetReader alloc] initWithAsset:asset error:&error];
            if (error || !reader) return false;

            // Configure for maximum performance with BGRA output
            NSDictionary* outputSettings = @{
                (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
                (id)kCVPixelBufferMetalCompatibilityKey: @YES,
                (id)kCVPixelBufferIOSurfacePropertiesKey: @{},
                // Add VideoToolbox hardware acceleration hints
                AVVideoDecompressionPropertiesKey: @{
                    (id)kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder: @YES,
                    (id)kVTDecompressionPropertyKey_PropagatePerFrameHDRDisplayMetadata: @NO,
                },
            };

            output = [[AVAssetReaderTrackOutput alloc]
                initWithTrack:videoTrack outputSettings:outputSettings];

            // Critical performance settings
            output.alwaysCopiesSampleData = NO;  // Avoid unnecessary copies
            output.supportsRandomAccess = YES;   // Enable seeking

            if (![reader canAddOutput:output]) {
                reader = nil;
                output = nil;
                return false;
            }

            [reader addOutput:output];

            // Set time range if seeking
            if (startFrame > 0) {
                CMTime startTime = CMTimeMake(startFrame, cachedFPS);
                CMTime duration = CMTimeSubtract([asset duration], startTime);
                reader.timeRange = CMTimeRangeMake(startTime, duration);
            }

            if (![reader startReading]) {
                reader = nil;
                output = nil;
                return false;
            }

            currentFrame = startFrame;
            batch_count = 0;
            batch_index = 0;
            return true;
        }
    }

    /// Load next batch of frames into internal buffer
    void loadBatch() {
        if (!reader || !output || !isOpen) {
            batch_count = 0;
            return;
        }

        size_t frame_size = cachedWidth * cachedHeight * 4;
        batch_count = 0;

        @autoreleasepool {
            while (batch_count < BATCH_SIZE) {
                if (reader.status != AVAssetReaderStatusReading) break;

                CMSampleBufferRef sampleBuffer = [output copyNextSampleBuffer];
                if (!sampleBuffer) break;

                CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
                if (imageBuffer) {
                    CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

                    uint8_t* src = (uint8_t*)CVPixelBufferGetBaseAddress(imageBuffer);
                    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
                    uint8_t* dst = batch_buffer.data() + (batch_count * frame_size);

                    if (bytesPerRow == cachedWidth * 4) {
                        memcpy(dst, src, frame_size);
                    } else {
                        size_t copy_width = cachedWidth * 4;
                        for (int y = 0; y < cachedHeight; y++) {
                            memcpy(dst + y * copy_width,
                                   src + y * bytesPerRow,
                                   copy_width);
                        }
                    }

                    CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
                    batch_count++;
                    currentFrame++;
                }

                CFRelease(sampleBuffer);
            }
        }

        batch_index = 0;
    }

    /// Get pointer to next frame from batch
    uint8_t* nextFrame() {
        if (!isOpen) return nullptr;

        // Load new batch if needed
        if (batch_index >= batch_count) {
            loadBatch();
            if (batch_count == 0) return nullptr;
        }

        size_t frame_size = cachedWidth * cachedHeight * 4;
        uint8_t* frame_ptr = batch_buffer.data() + (batch_index * frame_size);
        batch_index++;
        return frame_ptr;
    }

    void reset(int64_t frameIndex) {
        if (!isOpen) return;
        setupReader(frameIndex);
    }
};

// Public interface implementation
FrameExtractor::FrameExtractor() : impl(new Impl()) {}
FrameExtractor::~FrameExtractor() { delete impl; }

bool FrameExtractor::open(const std::string& path) {
    return impl->open(path);
}

uint8_t* FrameExtractor::next_frame() {
    return impl->nextFrame();
}

void FrameExtractor::reset(int64_t frame_index) {
    impl->reset(frame_index);
}

int FrameExtractor::width() const { return impl->cachedWidth; }
int FrameExtractor::height() const { return impl->cachedHeight; }
double FrameExtractor::fps() const { return impl->cachedFPS; }
int64_t FrameExtractor::total_frames() const { return impl->cachedTotalFrames; }

} // namespace viteo