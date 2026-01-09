#import <AVFoundation/AVFoundation.h>
#import <VideoToolbox/VideoToolbox.h>
#import <CoreVideo/CoreVideo.h>
#import <Accelerate/Accelerate.h>
#include "frame_extractor.h"
#include <iostream>
#include <vector>
#include <deque>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>

#define DEBUG_LOG(msg) do { \
    if (debugLogging) { \
        std::cerr << "[viteo] " << msg << std::endl; \
    } \
} while(0)

namespace viteo {

/// Internal implementation with AVFoundation and background prefetching
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

    // Threading / Prefetch state
    std::thread decodeThread;
    std::deque<std::shared_ptr<std::vector<uint8_t>>> frameQueue;
    std::mutex queueMutex;
    std::condition_variable producerCV;
    std::condition_variable consumerCV;

    static constexpr size_t MAX_QUEUE_SIZE = 4;
    std::atomic<bool> isRunning{false};
    std::atomic<bool> isFinished{false};
    std::atomic<bool> hasError{false};

    bool debugLogging = false;

    Impl() {
        if (std::getenv("VITEO_DEBUG")) {
            debugLogging = true;
        }
        DEBUG_LOG("Initialized frame extractor");
    }

    ~Impl() {
        close();
    }

    /// Releases all resources and resets state
    void close() {
        if (!asset && !reader) return;

        stopDecodeThread();

        @autoreleasepool {
            if (reader) {
                [reader cancelReading];
                reader = nil;
            }
            output = nil;
            videoTrack = nil;
            asset = nil;
        }
        DEBUG_LOG("Closed video resources");
    }

    /// Stops the background decode thread
    void stopDecodeThread() {
        isRunning = false;
        producerCV.notify_all();
        consumerCV.notify_all();

        if (decodeThread.joinable()) {
            decodeThread.join();
        }

        std::lock_guard<std::mutex> lock(queueMutex);
        frameQueue.clear();
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
        NSArray* tracks = [videoAsset tracksWithMediaType:AVMediaTypeVideo];

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
        numTotalFrames = static_cast<int64_t>(CMTimeGetSeconds(duration) * videoFPS);

        DEBUG_LOG("Video metadata: " << videoWidth << "x" << videoHeight
                  << " @ " << videoFPS << " fps, "
                  << numTotalFrames << " total frames");
    }

    /// Creates output settings for hardware-accelerated decoding
    NSDictionary* createOutputSettings() {
        return @{
            (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
            (id)kCVPixelBufferMetalCompatibilityKey: @YES,
            AVVideoDecompressionPropertiesKey: @{
                (id)kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder: @YES
            }
        };
    }

    /// Configures track output for optimal performance
    AVAssetReaderTrackOutput* createTrackOutput(AVAssetTrack* track, NSDictionary* settings) {
        AVAssetReaderTrackOutput* trackOutput = [[AVAssetReaderTrackOutput alloc]
            initWithTrack:track outputSettings:settings];

        trackOutput.alwaysCopiesSampleData = NO;

        DEBUG_LOG("Created track output with hardware acceleration");
        return trackOutput;
    }

    /// Initializes reader for frame extraction
    bool setupReader() {
        @autoreleasepool {
            if (reader) {
                [reader cancelReading];
                reader = nil;
                output = nil;
            }

            NSError* error = nil;
            reader = [[AVAssetReader alloc] initWithAsset:asset error:&error];
            if (error || !reader) {
                DEBUG_LOG("Failed to create reader");
                return false;
            }

            NSDictionary* settings = createOutputSettings();
            output = createTrackOutput(videoTrack, settings);

            if (![reader canAddOutput:output]) {
                DEBUG_LOG("Cannot add output to reader");
                reader = nil;
                output = nil;
                return false;
            }

            [reader addOutput:output];

            if (![reader startReading]) {
                DEBUG_LOG("Failed to start reading");
                reader = nil;
                output = nil;
                return false;
            }

            DEBUG_LOG("Reader initialized successfully");
            return true;
        }
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

            if (!setupReader()) return false;

            // Start background decode thread
            isRunning = true;
            isFinished = false;
            hasError = false;
            decodeThread = std::thread(&Impl::decodeLoop, this);

            DEBUG_LOG("Video opened successfully");
            return true;
        }
    }

    /// Copies pixel buffer data using vImage for optimal performance
    std::shared_ptr<std::vector<uint8_t>> copyPixelBuffer(CVImageBufferRef imageBuffer) {
        CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

        void* srcData = CVPixelBufferGetBaseAddress(imageBuffer);
        size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
        size_t width = CVPixelBufferGetWidth(imageBuffer);
        size_t height = CVPixelBufferGetHeight(imageBuffer);

        size_t dataSize = width * height * 4;
        auto buffer = std::make_shared<std::vector<uint8_t>>(dataSize);

        vImage_Buffer src = {
            .data = srcData,
            .height = static_cast<vImagePixelCount>(height),
            .width = static_cast<vImagePixelCount>(width),
            .rowBytes = bytesPerRow
        };

        vImage_Buffer dest = {
            .data = buffer->data(),
            .height = static_cast<vImagePixelCount>(height),
            .width = static_cast<vImagePixelCount>(width),
            .rowBytes = width * 4
        };

        vImage_Error err = vImageCopyBuffer(&src, &dest, 4, kvImageNoFlags);

        CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

        if (err != kvImageNoError) {
            DEBUG_LOG("vImage copy failed with error: " << err);
            return nullptr;
        }

        return buffer;
    }

    /// Background decode loop for frame prefetching
    void decodeLoop() {
        while (isRunning) {
            @autoreleasepool {
                std::unique_lock<std::mutex> lock(queueMutex);

                // Wait if queue is full
                producerCV.wait(lock, [this] {
                    return !isRunning || frameQueue.size() < MAX_QUEUE_SIZE;
                });

                if (!isRunning) break;
                lock.unlock();

                // Decode frame
                CMSampleBufferRef sample = [output copyNextSampleBuffer];

                if (!sample) {
                    std::lock_guard<std::mutex> guard(queueMutex);
                    isFinished = true;
                    consumerCV.notify_all();
                    DEBUG_LOG("End of stream reached");
                    break;
                }

                CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sample);
                std::shared_ptr<std::vector<uint8_t>> frameData = nullptr;

                if (imageBuffer) {
                    frameData = copyPixelBuffer(imageBuffer);
                }

                CFRelease(sample);

                if (frameData) {
                    std::lock_guard<std::mutex> guard(queueMutex);
                    frameQueue.push_back(frameData);
                    consumerCV.notify_one();
                } else {
                    hasError = true;
                    DEBUG_LOG("Failed to process sample buffer");
                    break;
                }
            }
        }
    }

    /// Returns next decoded frame
    std::shared_ptr<std::vector<uint8_t>> nextFrame() {
        std::unique_lock<std::mutex> lock(queueMutex);

        consumerCV.wait(lock, [this] {
            return !frameQueue.empty() || isFinished || !isRunning;
        });

        if (!frameQueue.empty()) {
            auto frame = frameQueue.front();
            frameQueue.pop_front();
            producerCV.notify_one();
            return frame;
        }

        DEBUG_LOG("No more frames available");
        return nullptr;
    }

    /// Resets to beginning (restarts decode pipeline)
    void reset(int64_t frameIndex) {
        if (!isRunning) return;
        DEBUG_LOG("Resetting to frame " << frameIndex);
        close();
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
