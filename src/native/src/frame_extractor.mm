#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>
#import <VideoToolbox/VideoToolbox.h>
#include "frame_extractor.h"
#include <atomic>
#include <dispatch/dispatch.h>

namespace viteo {

class FrameExtractor::Impl {
public:
    AVAsset* asset = nil;
    AVAssetReader* reader = nil;
    AVAssetReaderTrackOutput* output = nil;
    AVAssetTrack* videoTrack = nil;

    // Video properties cached
    int cachedWidth = 0;
    int cachedHeight = 0;
    double cachedFPS = 0.0;
    int64_t cachedTotalFrames = 0;
    int64_t currentFrame = 0;

    std::atomic<bool> isOpen{false};

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

            // Use the deprecated API for now as we need synchronous access
            #pragma clang diagnostic push
            #pragma clang diagnostic ignored "-Wdeprecated-declarations"
            NSArray* tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
            #pragma clang diagnostic pop

            if (tracks.count == 0) return false;

            videoTrack = tracks[0];

            // Cache properties
            CGSize size = [videoTrack naturalSize];
            cachedWidth = static_cast<int>(size.width);
            cachedHeight = static_cast<int>(size.height);
            cachedFPS = [videoTrack nominalFrameRate];

            // Estimate total frames
            CMTime duration = [asset duration];
            cachedTotalFrames = static_cast<int64_t>(
                CMTimeGetSeconds(duration) * cachedFPS
            );

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
            return true;
        }
    }

    size_t extractBatch(uint8_t* buffer, size_t batchSize) {
        if (!reader || !output || !isOpen) return 0;

        size_t framesExtracted = 0;
        size_t frameSize = cachedWidth * cachedHeight * 4;

        @autoreleasepool {
            while (framesExtracted < batchSize) {
                if (reader.status != AVAssetReaderStatusReading) {
                    break;
                }

                CMSampleBufferRef sampleBuffer = [output copyNextSampleBuffer];
                if (!sampleBuffer) {
                    break;
                }

                CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
                if (imageBuffer) {
                    // Lock with read-only flag for performance
                    CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

                    uint8_t* src = (uint8_t*)CVPixelBufferGetBaseAddress(imageBuffer);
                    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);

                    // Destination for this frame
                    uint8_t* dst = buffer + (framesExtracted * frameSize);

                    // Fast path: if stride matches, copy entire frame at once
                    if (bytesPerRow == cachedWidth * 4) {
                        memcpy(dst, src, frameSize);
                    } else {
                        // Row-by-row copy for padded buffers
                        size_t copyWidth = cachedWidth * 4;
                        for (int y = 0; y < cachedHeight; y++) {
                            memcpy(dst + y * copyWidth,
                                   src + y * bytesPerRow,
                                   copyWidth);
                        }
                    }

                    CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
                    framesExtracted++;
                    currentFrame++;
                }

                CFRelease(sampleBuffer);
            }
        }

        return framesExtracted;
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

int FrameExtractor::width() const { return impl->cachedWidth; }
int FrameExtractor::height() const { return impl->cachedHeight; }
double FrameExtractor::fps() const { return impl->cachedFPS; }
int64_t FrameExtractor::total_frames() const { return impl->cachedTotalFrames; }

size_t FrameExtractor::extract_batch(uint8_t* buffer, size_t batch_size) {
    return impl->extractBatch(buffer, batch_size);
}

void FrameExtractor::reset(int64_t frame_index) {
    impl->reset(frame_index);
}

} // namespace viteo