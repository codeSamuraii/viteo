#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>
#import <VideoToolbox/VideoToolbox.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreImage/CoreImage.h>
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

    ~Impl() {
        close();
    }

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
        if (!asset) {
            return false;
        }

        // Use synchronous load to avoid deprecation warning
        NSArray* tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
        if (tracks.count == 0) {
            return false;
        }

        videoTrack = tracks[0];
        return videoTrack != nil;
    }

    double getDuration() const {
        if (!asset) {
            return 0.0;
        }
        return CMTimeGetSeconds([asset duration]);
    }

    int getWidth() const {
        if (!videoTrack) {
            return 0;
        }
        CGSize size = [videoTrack naturalSize];
        return static_cast<int>(size.width);
    }

    int getHeight() const {
        if (!videoTrack) {
            return 0;
        }
        CGSize size = [videoTrack naturalSize];
        return static_cast<int>(size.height);
    }

    double getFPS() const {
        if (!videoTrack) {
            return 0.0;
        }
        return static_cast<double>([videoTrack nominalFrameRate]);
    }

    Frame convertSampleBufferToFrameFast(CMSampleBufferRef sampleBuffer, int64_t frameNumber) {
        Frame frame;
        frame.frame_number = frameNumber;

        CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        if (!imageBuffer) {
            return frame;
        }

        CMTime presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
        frame.timestamp = CMTimeGetSeconds(presentationTime);

        CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

        size_t width = CVPixelBufferGetWidth(imageBuffer);
        size_t height = CVPixelBufferGetHeight(imageBuffer);
        frame.width = static_cast<int>(width);
        frame.height = static_cast<int>(height);

        // Use BGRA directly - no conversion needed!
        uint8_t* baseAddress = (uint8_t*)CVPixelBufferGetBaseAddress(imageBuffer);
        size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
        size_t dataSize = bytesPerRow * height;

        // Fast memory copy (or ideally zero-copy if we can keep the buffer alive)
        frame.data.resize(dataSize);

        // Use memcpy for optimal performance
        if (bytesPerRow == width * 4) {
            // Contiguous memory - single copy
            memcpy(frame.data.data(), baseAddress, dataSize);
        } else {
            // Copy row by row if there's padding
            for (size_t y = 0; y < height; y++) {
                memcpy(frame.data.data() + y * width * 4,
                       baseAddress + y * bytesPerRow,
                       width * 4);
            }
        }

        CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

        return frame;
    }
};

// FrameExtractor implementation

FrameExtractor::FrameExtractor() : pImpl(new Impl()) {}

FrameExtractor::~FrameExtractor() {
    delete pImpl;
}

bool FrameExtractor::open(const std::string& path) {
    return pImpl->open(path);
}

void FrameExtractor::close() {
    pImpl->close();
}

double FrameExtractor::get_duration() const {
    return pImpl->getDuration();
}

int FrameExtractor::get_width() const {
    return pImpl->getWidth();
}

int FrameExtractor::get_height() const {
    return pImpl->getHeight();
}

double FrameExtractor::get_fps() const {
    return pImpl->getFPS();
}

bool FrameExtractor::start_streaming(double start_time, double end_time) {
    if (!pImpl->asset || !pImpl->videoTrack) {
        return false;
    }

    if (pImpl->reader) {
        [pImpl->reader cancelReading];
        pImpl->reader = nil;
        pImpl->output = nil;
    }

    @autoreleasepool {
        NSError* error = nil;

        pImpl->reader = [[AVAssetReader alloc] initWithAsset:pImpl->asset error:&error];
        if (error || !pImpl->reader) {
            return false;
        }

        // Configure output settings for hardware decoding
        NSDictionary* outputSettings = @{
            (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
            (id)kCVPixelBufferMetalCompatibilityKey: @YES,  // Metal-compatible for MLX
        };

        pImpl->output = [[AVAssetReaderTrackOutput alloc]
            initWithTrack:pImpl->videoTrack
            outputSettings:outputSettings];

        pImpl->output.alwaysCopiesSampleData = NO;  // Zero-copy when possible

        if (![pImpl->reader canAddOutput:pImpl->output]) {
            pImpl->reader = nil;
            pImpl->output = nil;
            return false;
        }

        [pImpl->reader addOutput:pImpl->output];

        if (start_time > 0 || end_time > 0) {
            CMTime start = CMTimeMakeWithSeconds(start_time, 600);
            CMTime duration;
            if (end_time > 0) {
                duration = CMTimeMakeWithSeconds(end_time - start_time, 600);
            } else {
                duration = CMTimeSubtract([pImpl->asset duration], start);
            }
            pImpl->reader.timeRange = CMTimeRangeMake(start, duration);
        }

        if (![pImpl->reader startReading]) {
            pImpl->reader = nil;
            pImpl->output = nil;
            return false;
        }

        pImpl->currentFrameNumber = 0;
        return true;
    }
}

size_t FrameExtractor::next_frames_batch(std::vector<Frame>& frames, size_t max_frames) {
    frames.clear();

    if (!pImpl->reader || !pImpl->output) {
        return 0;
    }

    if (pImpl->reader.status != AVAssetReaderStatusReading) {
        [pImpl->reader cancelReading];
        pImpl->reader = nil;
        pImpl->output = nil;
        return 0;
    }

    frames.reserve(max_frames);

    @autoreleasepool {
        for (size_t i = 0; i < max_frames; i++) {
            CMSampleBufferRef sampleBuffer = [pImpl->output copyNextSampleBuffer];

            if (!sampleBuffer) {
                // End of stream
                [pImpl->reader cancelReading];
                pImpl->reader = nil;
                pImpl->output = nil;
                break;
            }

            Frame frame = pImpl->convertSampleBufferToFrameFast(sampleBuffer, pImpl->currentFrameNumber++);
            CFRelease(sampleBuffer);

            if (!frame.data.empty()) {
                frames.push_back(std::move(frame));
            }
        }
    }

    return frames.size();
}

bool FrameExtractor::is_streaming() const {
    return pImpl->reader != nil && pImpl->reader.status == AVAssetReaderStatusReading;
}

} // namespace videoextractor
