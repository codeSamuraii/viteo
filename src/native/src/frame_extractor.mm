#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>
#import <VideoToolbox/VideoToolbox.h>
#include "frame_extractor.h"
#include <cstdlib>
#include <iostream>

#define DEBUG_LOG(msg) do { \
    if (std::getenv("VITEO_DEBUG")) { \
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

    int cachedWidth = 0;
    int cachedHeight = 0;
    double cachedFPS = 0.0;
    int64_t cachedTotalFrames = 0;
    int64_t currentFrame = 0;

    // Internal batch buffer for performance
    size_t batch_size;
    std::vector<uint8_t> batch_buffer;
    size_t batch_count = 0;
    size_t batch_index = 0;

    bool isOpen = false;

    Impl(size_t batch_size_param) : batch_size(batch_size_param) {
        DEBUG_LOG("Setting batch size to " << batch_size);
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
        cachedWidth = static_cast<int>(size.width);
        cachedHeight = static_cast<int>(size.height);
        cachedFPS = [track nominalFrameRate];

        CMTime duration = [videoAsset duration];
        cachedTotalFrames = static_cast<int64_t>(
            CMTimeGetSeconds(duration) * cachedFPS
        );

        DEBUG_LOG("Video metadata: " << cachedWidth << "x" << cachedHeight
                  << " @ " << cachedFPS << " fps, "
                  << cachedTotalFrames << " total frames");
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

            // Allocate batch buffer
            size_t frame_size = cachedWidth * cachedHeight * 4;
            batch_buffer.resize(batch_size * frame_size);
            DEBUG_LOG("Allocated batch buffer for " << batch_size << " frames");

            isOpen = true;
            return setupReader(0);
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
            CMTime startTime = CMTimeMake(startFrame, cachedFPS);
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
            batch_count = 0;
            batch_index = 0;
            DEBUG_LOG("Reader initialized successfully");
            return true;
        }
    }

    /// Copies frame from pixel buffer to destination
    void copyFrameData(CVImageBufferRef imageBuffer, uint8_t* dst) {
        uint8_t* src = (uint8_t*)CVPixelBufferGetBaseAddress(imageBuffer);
        size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
        size_t data_width = cachedWidth * 4;
        size_t data_size = cachedHeight * data_width;

        if (bytesPerRow == data_width) {
            memcpy(dst, src, data_size);
        } else {

            for (int y = 0; y < cachedHeight; y++) {
                memcpy(dst + y * data_width,
                       src + y * bytesPerRow,
                       data_width);
            }
        }
    }

    /// Processes single sample buffer and adds to batch
    bool processSampleBuffer(CMSampleBufferRef sampleBuffer, size_t frame_size) {
        CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        if (!imageBuffer) return false;

        CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

        uint8_t* dst = batch_buffer.data() + (batch_count * frame_size);
        copyFrameData(imageBuffer, dst);

        CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
        batch_count++;
        currentFrame++;

        return true;
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
            while (batch_count < batch_size) {
                if (reader.status != AVAssetReaderStatusReading) {
                    DEBUG_LOG("Reader stopped, loaded " << batch_count << " frames");
                    break;
                }

                CMSampleBufferRef sampleBuffer = [output copyNextSampleBuffer];
                if (!sampleBuffer) {
                    DEBUG_LOG("No more sample buffers, loaded " << batch_count << " frames");
                    break;
                }

                processSampleBuffer(sampleBuffer, frame_size);
                CFRelease(sampleBuffer);
            }
        }

        batch_index = 0;
        if (batch_count > 0) {
            DEBUG_LOG("Loaded batch of " << batch_count << " frames");
        }
    }

    /// Returns pointer to next frame from batch
    uint8_t* nextFrame() {
        if (!isOpen) return nullptr;

        if (batch_index >= batch_count) {
            loadBatch();
            if (batch_count == 0) {
                DEBUG_LOG("No more frames available");
                return nullptr;
            }
        }

        size_t frame_size = cachedWidth * cachedHeight * 4;
        uint8_t* frame_ptr = batch_buffer.data() + (batch_index * frame_size);
        batch_index++;
        return frame_ptr;
    }

    /// Resets reader to specified frame index
    void reset(int64_t frameIndex) {
        if (!isOpen) return;
        DEBUG_LOG("Resetting to frame " << frameIndex);
        setupReader(frameIndex);
    }
};

// Public interface implementation
FrameExtractor::FrameExtractor(size_t batch_size_param) : impl(new Impl(batch_size_param)) {}
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