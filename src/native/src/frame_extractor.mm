#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>
#import <VideoToolbox/VideoToolbox.h>
#import <Accelerate/Accelerate.h>
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
    int numChannels = 4;
    double videoFPS = 0.0;
    int64_t numTotalFrames = 0;
    int64_t currentFrame = 0;

    std::vector<uint8_t> frame_buffer;
    std::vector<uint8_t> y_buffer;
    std::vector<uint8_t> uv_buffer;

    vImage_YpCbCrToARGB conversionInfo;
    bool conversionReady = false;
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
            conversionReady = false;
            currentFrame = 0;
        }
        frame_buffer.clear(); frame_buffer.shrink_to_fit();
        y_buffer.clear(); y_buffer.shrink_to_fit();
        uv_buffer.clear(); uv_buffer.shrink_to_fit();
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

        CMTime duration = track.timeRange.duration;
        numTotalFrames = std::llround(
            CMTimeGetSeconds(duration) * videoFPS
        );

        DEBUG_LOG("Video metadata: " << videoWidth << "x" << videoHeight
                  << " @ " << videoFPS << " fps, "
                  << numTotalFrames << " total frames");
    }

    /// Initializes vImage YUV-to-ARGB conversion (called once per open)
    bool initConversionInfo() {
        vImage_YpCbCrPixelRange pixelRange = {
            .Yp_bias = 16,
            .CbCr_bias = 128,
            .YpRangeMax = 235,
            .CbCrRangeMax = 240,
            .YpMax = 235,
            .YpMin = 16,
            .CbCrMax = 240,
            .CbCrMin = 16
        };

        vImage_Error err = vImageConvert_YpCbCrToARGB_GenerateConversion(
            kvImage_YpCbCrToARGBMatrix_ITU_R_709_2,
            &pixelRange,
            &conversionInfo,
            kvImage420Yp8_CbCr8,
            kvImageARGB8888,
            kvImageNoFlags
        );

        conversionReady = (err == kvImageNoError);
        if (!conversionReady) {
            DEBUG_LOG("Failed to initialize vImage conversion: " << err);
        } else {
            DEBUG_LOG("vImage BT.709 conversion initialized");
        }
        return conversionReady;
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

            // Allocate output buffer (BGRA)
            size_t frameSize = videoWidth * videoHeight * numChannels;
            frame_buffer.resize(frameSize);

            // Allocate intermediate NV12 plane buffers
            y_buffer.resize(videoWidth * videoHeight);
            uv_buffer.resize(videoWidth * (videoHeight / 2));

            DEBUG_LOG("Allocated buffers: output=" << (frameSize / 1024 / 1024)
                      << " MB, Y=" << (y_buffer.size() / 1024 / 1024)
                      << " MB, UV=" << (uv_buffer.size() / 1024 / 1024) << " MB");

            if (!initConversionInfo()) return false;

            isOpen = true;
            if (!setupReader(0)) return false;

            DEBUG_LOG("Video opened successfully");
            return true;
        }
    }

    /// Creates output settings for native NV12 decode
    NSDictionary* createOutputSettings() {
        return @{
            (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
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

    /// Copies a plane from CVPixelBuffer into a contiguous buffer
    void copyPlane(CVImageBufferRef imageBuffer, int planeIndex,
                   uint8_t* dst, size_t dstRowBytes, int planeHeight) {
        uint8_t* src = (uint8_t*)CVPixelBufferGetBaseAddressOfPlane(imageBuffer, planeIndex);
        size_t srcRowBytes = CVPixelBufferGetBytesPerRowOfPlane(imageBuffer, planeIndex);

        if (srcRowBytes == dstRowBytes) {
            memcpy(dst, src, dstRowBytes * planeHeight);
        } else {
            for (int y = 0; y < planeHeight; y++) {
                memcpy(dst + y * dstRowBytes, src + y * srcRowBytes, dstRowBytes);
            }
        }
    }

    /// Converts NV12 pixel buffer to BGRA using vImage
    void convertFrame(CVImageBufferRef imageBuffer, uint8_t* dst) {
        // Copy Y plane (full resolution)
        copyPlane(imageBuffer, 0, y_buffer.data(), videoWidth, videoHeight);

        // Copy UV plane (half height, full width for interleaved CbCr)
        int uvHeight = videoHeight / 2;
        copyPlane(imageBuffer, 1, uv_buffer.data(), videoWidth, uvHeight);

        // Set up vImage buffers
        vImage_Buffer srcY = {
            .data = y_buffer.data(),
            .height = static_cast<vImagePixelCount>(videoHeight),
            .width = static_cast<vImagePixelCount>(videoWidth),
            .rowBytes = static_cast<size_t>(videoWidth)
        };

        vImage_Buffer srcUV = {
            .data = uv_buffer.data(),
            .height = static_cast<vImagePixelCount>(uvHeight),
            .width = static_cast<vImagePixelCount>(videoWidth / 2),
            .rowBytes = static_cast<size_t>(videoWidth)
        };

        vImage_Buffer dstBuf = {
            .data = dst,
            .height = static_cast<vImagePixelCount>(videoHeight),
            .width = static_cast<vImagePixelCount>(videoWidth),
            .rowBytes = static_cast<size_t>(videoWidth * numChannels)
        };

        // Permute ARGB → BGRA in a single pass
        uint8_t permuteMap[4] = {3, 2, 1, 0};

        vImageConvert_420Yp8_CbCr8ToARGB8888(
            &srcY, &srcUV, &dstBuf, &conversionInfo,
            permuteMap, 255, kvImageNoFlags
        );
    }

    uint8_t* nextFrame() {
        if (!isOpen || !reader || !output || !conversionReady) {
            DEBUG_LOG("Not ready to extract frames");
            return nullptr;
        }

        CMSampleBufferRef sampleBuffer = [output copyNextSampleBuffer];
        if (!sampleBuffer) {
            DEBUG_LOG("No more samples (reader status: " << reader.status << ")");
            return nullptr;
        }

        CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        if (!imageBuffer) {
            CFRelease(sampleBuffer);
            DEBUG_LOG("Failed to get image buffer from sample");
            return nullptr;
        }

        CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
        convertFrame(imageBuffer, frame_buffer.data());
        CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

        CFRelease(sampleBuffer);
        DEBUG_LOG("Returning frame " << currentFrame);
        currentFrame++;

        return frame_buffer.data();
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

uint8_t* FrameExtractor::next_frame() {
    return impl->nextFrame();
}

void FrameExtractor::reset(int64_t frame_index) {
    impl->reset(frame_index);
}

int FrameExtractor::width() const { return impl->videoWidth; }
int FrameExtractor::height() const { return impl->videoHeight; }
int FrameExtractor::channels() const { return impl->numChannels; }
double FrameExtractor::fps() const { return impl->videoFPS; }
int64_t FrameExtractor::total_frames() const { return impl->numTotalFrames; }

} // namespace viteo
