#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>
#import <VideoToolbox/VideoToolbox.h>
#include "frame_extractor.h"
#include <cstdlib>
#include <cmath>
#include <iostream>

#define DEBUG_LOG(msg) do { \
    if (debugLogging) { \
        std::cerr << "[viteo] " << msg << std::endl; \
    } \
} while(0)

namespace viteo {

/// Internal implementation with AVFoundation
class FrameExtractor::Impl {
public:
    AVAsset* asset = nil;
    AVAssetReader* reader = nil;
    AVAssetReaderTrackOutput* output = nil;
    AVAssetTrack* videoTrack = nil;

    int videoWidth = 0;
    int videoHeight = 0;
    double videoFPS = 0.0;
    int64_t numTotalFrames = 0;
    int64_t currentFrame = 0;

    std::shared_ptr<std::vector<uint8_t>> frame_buffer;

    bool isOpen = false;
    bool debugLogging = false;

    Impl() {
        if (std::getenv("VITEO_DEBUG")) {
            debugLogging = true;
        }
        DEBUG_LOG("Initialized frame extractor");
    }

    ~Impl() {
        close();
        // ARC handles cleanup automatically
    }

    /// Releases all resources and resets state
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
        DEBUG_LOG("Closed video resources");
    }

    /// Loads asset from file path
    AVAsset* loadAsset(const std::string& path) {
        NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:nsPath];
        AVAsset* loadedAsset = [AVAsset assetWithURL:url];

        if (loadedAsset) {
            DEBUG_LOG("Loaded asset from: " << path);
        } else {
            DEBUG_LOG("Failed to load asset from: " << path);
        }

        return loadedAsset;
    }

    /// Extracts video track from asset
    AVAssetTrack* extractVideoTrack(AVAsset* videoAsset) {
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        NSArray* tracks = [videoAsset tracksWithMediaType:AVMediaTypeVideo];
        #pragma clang diagnostic pop

        if (tracks.count == 0) {
            DEBUG_LOG("No video tracks found");
            return nil;
        }

        DEBUG_LOG("Found " << tracks.count << " video track(s)");
        return tracks[0];
    }

    /// Caches video metadata from track
    void cacheMetadata(AVAssetTrack* track, AVAsset* videoAsset) {
        CGSize size = [track naturalSize];
        videoWidth = static_cast<int>(size.width);
        videoHeight = static_cast<int>(size.height);
        videoFPS = [track nominalFrameRate];

        CMTime duration = [videoAsset duration];
        numTotalFrames = std::llround(
            CMTimeGetSeconds(duration) * videoFPS
        );

        DEBUG_LOG("Video metadata: " << videoWidth << "x" << videoHeight
                  << " @ " << videoFPS << " fps, "
                  << numTotalFrames << " total frames");
    }

    /// Opens video file and initializes extraction
    bool open(const std::string& path) {
        close();

        @autoreleasepool {
            asset = loadAsset(path);
            if (!asset) return false;

            videoTrack = extractVideoTrack(asset);
            if (!videoTrack) return false;

            cacheMetadata(videoTrack, asset);

            size_t frameSize = videoWidth * videoHeight * 4;
            frame_buffer = std::make_shared<std::vector<uint8_t>>(frameSize);
            DEBUG_LOG("Allocated frame buffer (" << (frameSize / 1024 / 1024) << " MB)");

            isOpen = true;
            if (!setupReader(0)) return false;

            DEBUG_LOG("Video opened successfully");
            return true;
        }
    }

    /// Creates output settings dictionary for hardware accelerated decoding
    NSDictionary* createOutputSettings() {
        return @{
            (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
            (id)kCVPixelBufferMetalCompatibilityKey: @YES,
            (id)kCVPixelBufferIOSurfacePropertiesKey: @{},
            AVVideoDecompressionPropertiesKey: @{
                (id)kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder: @YES,
                (id)kVTDecompressionPropertyKey_PropagatePerFrameHDRDisplayMetadata: @NO,
            },
        };
    }

    /// Configures track output for optimal performance
    AVAssetReaderTrackOutput* createTrackOutput(AVAssetTrack* track, NSDictionary* settings) {
        AVAssetReaderTrackOutput* trackOutput = [[AVAssetReaderTrackOutput alloc]
            initWithTrack:track outputSettings:settings];

        trackOutput.alwaysCopiesSampleData = NO;
        trackOutput.supportsRandomAccess = YES;

        DEBUG_LOG("Created track output with hardware acceleration");
        return trackOutput;
    }

    /// Applies time range for seeking to specific frame
    void applyTimeRange(AVAssetReader* videoReader, int64_t startFrame) {
        if (startFrame > 0) {
            CMTime startTime = CMTimeMake(startFrame, videoFPS);
            CMTime duration = CMTimeSubtract([asset duration], startTime);
            videoReader.timeRange = CMTimeRangeMake(startTime, duration);
            DEBUG_LOG("Seeking to frame " << startFrame);
        }
    }

    /// Initializes reader for frame extraction
    bool setupReader(int64_t startFrame) {
        @autoreleasepool {
            if (reader) {
                [reader cancelReading];
                reader = nil;
                output = nil;
            }

            NSError* error = nil;
            reader = [[AVAssetReader alloc] initWithAsset:asset error:&error];
            if (error || !reader) {
                DEBUG_LOG("Failed to create reader: " << (error ? [[error localizedDescription] UTF8String] : "unknown error"));
                return false;
            }

            NSDictionary* outputSettings = createOutputSettings();
            output = createTrackOutput(videoTrack, outputSettings);

            if (![reader canAddOutput:output]) {
                DEBUG_LOG("Cannot add output to reader");
                reader = nil;
                output = nil;
                return false;
            }

            [reader addOutput:output];
            applyTimeRange(reader, startFrame);

            if (![reader startReading]) {
                DEBUG_LOG("Failed to start reading");
                reader = nil;
                output = nil;
                return false;
            }

            currentFrame = startFrame;
            DEBUG_LOG("Reader initialized successfully");
            return true;
        }
    }

    /// Copies frame from pixel buffer to destination
    void copyFrameData(CVImageBufferRef imageBuffer, uint8_t* dst) {
        uint8_t* src = (uint8_t*)CVPixelBufferGetBaseAddress(imageBuffer);
        size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
        size_t dataWidth = videoWidth * 4;
        size_t dataSize = videoHeight * dataWidth;

        if (bytesPerRow == dataWidth) {
            memcpy(dst, src, dataSize);
        } else {
            for (int y = 0; y < videoHeight; y++) {
                memcpy(dst + y * dataWidth,
                       src + y * bytesPerRow,
                       dataWidth);
            }
        }
    }

    std::shared_ptr<std::vector<uint8_t>> nextFrame() {
        if (!isOpen || !reader || !output) {
            DEBUG_LOG("Not ready to extract frames");
            return nullptr;
        }

        @autoreleasepool {
            if (frame_buffer.use_count() > 1 || !frame_buffer) {
                size_t frameSize = static_cast<size_t>(videoWidth) * static_cast<size_t>(videoHeight) * 4;
                frame_buffer = std::make_shared<std::vector<uint8_t>>(frameSize);
                DEBUG_LOG("Allocated fresh frame buffer due to active references");
            }

            if (reader.status != AVAssetReaderStatusReading) {
                DEBUG_LOG("Reader not in reading state");
                return nullptr;
            }

            CMSampleBufferRef sampleBuffer = [output copyNextSampleBuffer];
            if (!sampleBuffer) {
                DEBUG_LOG("No more samples available");
                return nullptr;
            }

            CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
            if (!imageBuffer) {
                CFRelease(sampleBuffer);
                DEBUG_LOG("Failed to get image buffer from sample");
                return nullptr;
            }

            CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
            copyFrameData(imageBuffer, frame_buffer->data());
            CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

            CFRelease(sampleBuffer);
            DEBUG_LOG("Returning frame " << currentFrame);
            currentFrame++;

            return frame_buffer;
        }
    }

    void reset(int64_t frameIndex) {
        if (!isOpen) return;
        DEBUG_LOG("Resetting to frame " << frameIndex);
        setupReader(frameIndex);
    }
};

// Public interface implementation
FrameExtractor::FrameExtractor() : impl(new Impl()) {}
FrameExtractor::~FrameExtractor() { delete impl; }

bool FrameExtractor::open(const std::string& path) {
    return impl->open(path);
}

std::shared_ptr<std::vector<uint8_t>> FrameExtractor::next_frame() {
    return impl->nextFrame();
}

void FrameExtractor::reset(int64_t frame_index) {
    impl->reset(frame_index);
}

int FrameExtractor::width() const { return impl->videoWidth; }
int FrameExtractor::height() const { return impl->videoHeight; }
double FrameExtractor::fps() const { return impl->videoFPS; }
int64_t FrameExtractor::total_frames() const { return impl->numTotalFrames; }

} // namespace viteo