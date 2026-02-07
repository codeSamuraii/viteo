#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>
#import <VideoToolbox/VideoToolbox.h>
#import <Accelerate/Accelerate.h>
#include "frame_extractor.h"
#include <cstdlib>
#include <cmath>
#include <iostream>

// Level 1: lifecycle and configuration events (open, close, reset, errors)
// Level 2: per-frame decode loop details (sample buffers, plane copies, conversion)
#define LOG1(msg) do { if (debugLevel >= 1) std::cerr << "[viteo] " << msg << std::endl; } while(0)
#define LOG2(msg) do { if (debugLevel >= 2) std::cerr << "[viteo] " << msg << std::endl; } while(0)

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
    int debugLevel = 0;

    Impl() {
        const char* env = std::getenv("VITEO_DEBUG");
        if (env) {
            debugLevel = std::atoi(env);
            if (debugLevel < 1) debugLevel = 1;
        }
        LOG1("Initialized extractor (debug level " << debugLevel << ")");
    }

    ~Impl() {
        close();
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
            conversionReady = false;
            currentFrame = 0;
        }
        frame_buffer.clear(); frame_buffer.shrink_to_fit();
        y_buffer.clear(); y_buffer.shrink_to_fit();
        uv_buffer.clear(); uv_buffer.shrink_to_fit();
        LOG1("Closed and released all resources");
    }

    AVAsset* loadAsset(const std::string& path) {
        LOG1("Loading asset from " << path);
        NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:nsPath];
        AVAsset* loadedAsset = [AVAsset assetWithURL:url];

        if (!loadedAsset) {
            LOG1("Failed to create AVAsset from path");
        }

        return loadedAsset;
    }

    AVAssetTrack* extractVideoTrack(AVAsset* videoAsset) {
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        NSArray* tracks = [videoAsset tracksWithMediaType:AVMediaTypeVideo];
        #pragma clang diagnostic pop

        if (tracks.count == 0) {
            LOG1("No video tracks found in asset");
            return nil;
        }

        LOG1("Found " << tracks.count << " video track(s), using first");
        return tracks[0];
    }

    void cacheMetadata(AVAssetTrack* track, AVAsset* videoAsset) {
        CGSize size = [track naturalSize];
        videoWidth = static_cast<int>(size.width);
        videoHeight = static_cast<int>(size.height);
        videoFPS = [track nominalFrameRate];

        CMTime duration = track.timeRange.duration;
        double durationSec = CMTimeGetSeconds(duration);
        numTotalFrames = std::llround(durationSec * videoFPS);

        LOG1("Track: " << videoWidth << "x" << videoHeight
             << ", " << videoFPS << " fps"
             << ", " << durationSec << "s"
             << ", ~" << numTotalFrames << " frames");
    }

    bool initConversionInfo() {
        LOG1("Initializing vImage NV12->BGRA conversion (BT.709 video range)");

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
            LOG1("vImage conversion init failed (error " << err << ")");
        } else {
            LOG1("vImage conversion ready");
        }
        return conversionReady;
    }

    bool open(const std::string& path) {
        close();

        @autoreleasepool {
            asset = loadAsset(path);
            if (!asset) return false;

            videoTrack = extractVideoTrack(asset);
            if (!videoTrack) return false;

            cacheMetadata(videoTrack, asset);

            size_t frameSize = videoWidth * videoHeight * numChannels;
            frame_buffer.resize(frameSize);
            y_buffer.resize(videoWidth * videoHeight);
            uv_buffer.resize(videoWidth * (videoHeight / 2));

            LOG1("Buffers allocated: BGRA=" << (frameSize / 1024) << "K"
                 << ", Y=" << (y_buffer.size() / 1024) << "K"
                 << ", UV=" << (uv_buffer.size() / 1024) << "K");

            if (!initConversionInfo()) return false;

            isOpen = true;
            if (!setupReader(0)) return false;

            LOG1("Open complete, ready to decode");
            return true;
        }
    }

    NSDictionary* createOutputSettings() {
        LOG1("Requesting NV12 output (420YpCbCr8BiPlanarVideoRange) with hardware decode");
        return @{
            (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
            AVVideoDecompressionPropertiesKey: @{
                (id)kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder: @YES,
                (id)kVTDecompressionPropertyKey_PropagatePerFrameHDRDisplayMetadata: @NO,
            },
        };
    }

    AVAssetReaderTrackOutput* createTrackOutput(AVAssetTrack* track, NSDictionary* settings) {
        AVAssetReaderTrackOutput* trackOutput = [[AVAssetReaderTrackOutput alloc]
            initWithTrack:track outputSettings:settings];

        trackOutput.alwaysCopiesSampleData = NO;
        trackOutput.supportsRandomAccess = YES;

        LOG1("Track output created (alwaysCopiesSampleData=NO, supportsRandomAccess=YES)");
        return trackOutput;
    }

    void applyTimeRange(AVAssetReader* videoReader, int64_t startFrame) {
        if (startFrame > 0) {
            CMTime startTime = CMTimeMake(startFrame, videoFPS);
            CMTime duration = CMTimeSubtract([asset duration], startTime);
            videoReader.timeRange = CMTimeRangeMake(startTime, duration);
            LOG1("Time range set: start frame " << startFrame
                 << " (" << CMTimeGetSeconds(startTime) << "s)");
        }
    }

    bool setupReader(int64_t startFrame) {
        LOG1("Setting up AVAssetReader (start frame " << startFrame << ")");

        @autoreleasepool {
            if (reader) {
                LOG1("Cancelling previous reader");
                [reader cancelReading];
                reader = nil;
                output = nil;
            }

            NSError* error = nil;
            reader = [[AVAssetReader alloc] initWithAsset:asset error:&error];
            if (error || !reader) {
                LOG1("AVAssetReader creation failed: "
                     << (error ? [[error localizedDescription] UTF8String] : "unknown"));
                return false;
            }

            NSDictionary* outputSettings = createOutputSettings();
            output = createTrackOutput(videoTrack, outputSettings);

            if (![reader canAddOutput:output]) {
                LOG1("Reader rejected track output");
                reader = nil;
                output = nil;
                return false;
            }

            [reader addOutput:output];
            applyTimeRange(reader, startFrame);

            if (![reader startReading]) {
                LOG1("Reader failed to start");
                reader = nil;
                output = nil;
                return false;
            }

            currentFrame = startFrame;
            LOG1("Reader started, decoding from frame " << startFrame);
            return true;
        }
    }

    void copyPlane(CVImageBufferRef imageBuffer, int planeIndex,
                   uint8_t* dst, size_t dstRowBytes, int planeHeight) {
        uint8_t* src = (uint8_t*)CVPixelBufferGetBaseAddressOfPlane(imageBuffer, planeIndex);
        size_t srcRowBytes = CVPixelBufferGetBytesPerRowOfPlane(imageBuffer, planeIndex);

        if (srcRowBytes == dstRowBytes) {
            memcpy(dst, src, dstRowBytes * planeHeight);
            LOG2("Plane " << planeIndex << ": bulk copy "
                 << (dstRowBytes * planeHeight / 1024) << "K");
        } else {
            for (int y = 0; y < planeHeight; y++) {
                memcpy(dst + y * dstRowBytes, src + y * srcRowBytes, dstRowBytes);
            }
            LOG2("Plane " << planeIndex << ": strided copy "
                 << planeHeight << " rows"
                 << " (src stride " << srcRowBytes
                 << ", dst stride " << dstRowBytes << ")");
        }
    }

    void convertFrame(CVImageBufferRef imageBuffer, uint8_t* dst) {
        LOG2("Copying Y plane (" << videoWidth << "x" << videoHeight << ")");
        copyPlane(imageBuffer, 0, y_buffer.data(), videoWidth, videoHeight);

        int uvHeight = videoHeight / 2;
        LOG2("Copying UV plane (" << videoWidth << "x" << uvHeight << ")");
        copyPlane(imageBuffer, 1, uv_buffer.data(), videoWidth, uvHeight);

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

        uint8_t permuteMap[4] = {3, 2, 1, 0};

        LOG2("vImage NV12->BGRA conversion (" << videoWidth << "x" << videoHeight << ")");
        vImageConvert_420Yp8_CbCr8ToARGB8888(
            &srcY, &srcUV, &dstBuf, &conversionInfo,
            permuteMap, 255, kvImageNoFlags
        );
    }

    uint8_t* nextFrame() {
        if (!isOpen || !reader || !output || !conversionReady) {
            LOG1("nextFrame called but not ready");
            return nullptr;
        }

        LOG2("Pulling sample buffer for frame " << currentFrame);
        CMSampleBufferRef sampleBuffer = [output copyNextSampleBuffer];
        if (!sampleBuffer) {
            LOG1("End of stream at frame " << currentFrame
                 << " (reader status " << reader.status << ")");
            return nullptr;
        }

        CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        if (!imageBuffer) {
            CFRelease(sampleBuffer);
            LOG1("Null image buffer at frame " << currentFrame);
            return nullptr;
        }

        LOG2("Locking pixel buffer");
        CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
        convertFrame(imageBuffer, frame_buffer.data());
        CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
        LOG2("Pixel buffer unlocked");

        CFRelease(sampleBuffer);
        LOG2("Frame " << currentFrame << " decoded");
        currentFrame++;

        return frame_buffer.data();
    }

    void reset(int64_t frameIndex) {
        if (!isOpen) return;
        LOG1("Reset to frame " << frameIndex);
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
