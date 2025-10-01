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

    Impl() : asset(nil), reader(nil), output(nil), videoTrack(nil) {}

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
    }

    bool open(const std::string& path) {
        close();

        NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:nsPath];

        asset = [AVAsset assetWithURL:url];
        if (!asset) {
            return false;
        }

        NSArray* tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
        if (tracks.count == 0) {
            return false;
        }

        videoTrack = tracks[0];
        return videoTrack != nil;
    }

    Frame extractFrameAtTime(double timestamp) {
        Frame frame;

        if (!asset || !videoTrack) {
            return frame;
        }

        @autoreleasepool {
            CMTime time = CMTimeMakeWithSeconds(timestamp, 600);

            AVAssetImageGenerator* generator = [[AVAssetImageGenerator alloc] initWithAsset:asset];
            generator.requestedTimeToleranceBefore = kCMTimeZero;
            generator.requestedTimeToleranceAfter = kCMTimeZero;
            generator.appliesPreferredTrackTransform = YES;

            // Request hardware acceleration
            generator.apertureMode = AVAssetImageGeneratorApertureModeCleanAperture;

            NSError* error = nil;
            CMTime actualTime;
            CGImageRef imageRef = [generator copyCGImageAtTime:time actualTime:&actualTime error:&error];

            if (error || !imageRef) {
                if (imageRef) {
                    CGImageRelease(imageRef);
                }
                return frame;
            }

            // Get image properties
            size_t width = CGImageGetWidth(imageRef);
            size_t height = CGImageGetHeight(imageRef);

            // Create RGB buffer
            std::vector<uint8_t> rgbData(width * height * 3);

            // Create context and draw image
            CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
            CGContextRef context = CGBitmapContextCreate(
                rgbData.data(),
                width,
                height,
                8,
                width * 3,
                colorSpace,
                kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault
            );

            if (context) {
                CGContextDrawImage(context, CGRectMake(0, 0, width, height), imageRef);

                frame.data = std::move(rgbData);
                frame.width = static_cast<int>(width);
                frame.height = static_cast<int>(height);
                frame.timestamp = CMTimeGetSeconds(actualTime);

                CGContextRelease(context);
            }

            CGColorSpaceRelease(colorSpace);
            CGImageRelease(imageRef);
        }

        return frame;
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

    Frame convertSampleBufferToFrame(CMSampleBufferRef sampleBuffer, int64_t frameNumber) {
        Frame frame;
        frame.frame_number = frameNumber;

        CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        if (!imageBuffer) {
            return frame;
        }

        // Get timestamp
        CMTime presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
        frame.timestamp = CMTimeGetSeconds(presentationTime);

        // Lock the pixel buffer
        CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

        size_t width = CVPixelBufferGetWidth(imageBuffer);
        size_t height = CVPixelBufferGetHeight(imageBuffer);
        frame.width = static_cast<int>(width);
        frame.height = static_cast<int>(height);

        // Get pixel format
        OSType pixelFormat = CVPixelBufferGetPixelFormatType(imageBuffer);

        // Allocate RGB buffer
        frame.data.resize(width * height * 3);

        // Convert based on pixel format
        if (pixelFormat == kCVPixelFormatType_32BGRA || pixelFormat == kCVPixelFormatType_32ARGB) {
            // Handle BGRA/ARGB format
            uint8_t* baseAddress = (uint8_t*)CVPixelBufferGetBaseAddress(imageBuffer);
            size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);

            for (size_t y = 0; y < height; y++) {
                uint8_t* rowPtr = baseAddress + y * bytesPerRow;
                for (size_t x = 0; x < width; x++) {
                    size_t srcIdx = x * 4;
                    size_t dstIdx = (y * width + x) * 3;

                    if (pixelFormat == kCVPixelFormatType_32BGRA) {
                        frame.data[dstIdx + 0] = rowPtr[srcIdx + 2]; // R
                        frame.data[dstIdx + 1] = rowPtr[srcIdx + 1]; // G
                        frame.data[dstIdx + 2] = rowPtr[srcIdx + 0]; // B
                    } else {
                        frame.data[dstIdx + 0] = rowPtr[srcIdx + 1]; // R
                        frame.data[dstIdx + 1] = rowPtr[srcIdx + 2]; // G
                        frame.data[dstIdx + 2] = rowPtr[srcIdx + 3]; // B
                    }
                }
            }
        } else {
            // For other formats, use CoreGraphics conversion
            CIImage* ciImage = [CIImage imageWithCVPixelBuffer:imageBuffer];
            CIContext* context = [CIContext contextWithOptions:nil];

            CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
            CGImageRef cgImage = [context createCGImage:ciImage fromRect:CGRectMake(0, 0, width, height)];

            if (cgImage) {
                CGContextRef bitmapContext = CGBitmapContextCreate(
                    frame.data.data(),
                    width,
                    height,
                    8,
                    width * 3,
                    colorSpace,
                    kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault
                );

                if (bitmapContext) {
                    CGContextDrawImage(bitmapContext, CGRectMake(0, 0, width, height), cgImage);
                    CGContextRelease(bitmapContext);
                }

                CGImageRelease(cgImage);
            }

            CGColorSpaceRelease(colorSpace);
        }

        CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

        return frame;
    }

    void streamFrames(double startTime, double endTime, FrameCallback callback) {
        if (!asset || !videoTrack) {
            return;
        }

        @autoreleasepool {
            NSError* error = nil;

            // Create asset reader
            AVAssetReader* streamReader = [[AVAssetReader alloc] initWithAsset:asset error:&error];
            if (error || !streamReader) {
                return;
            }

            // Configure output settings for hardware decoding
            NSDictionary* outputSettings = @{
                (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
                (id)kCVPixelBufferMetalCompatibilityKey: @YES,  // Enable Metal compatibility for GPU
            };

            AVAssetReaderTrackOutput* trackOutput = [[AVAssetReaderTrackOutput alloc]
                initWithTrack:videoTrack
                outputSettings:outputSettings];

            trackOutput.alwaysCopiesSampleData = NO;  // Zero-copy when possible

            if (![streamReader canAddOutput:trackOutput]) {
                return;
            }

            [streamReader addOutput:trackOutput];

            // Set time range if specified
            if (startTime > 0 || endTime > 0) {
                CMTime start = CMTimeMakeWithSeconds(startTime, 600);
                CMTime duration;
                if (endTime > 0) {
                    duration = CMTimeMakeWithSeconds(endTime - startTime, 600);
                } else {
                    duration = CMTimeSubtract([asset duration], start);
                }
                streamReader.timeRange = CMTimeRangeMake(start, duration);
            }

            // Start reading
            if (![streamReader startReading]) {
                return;
            }

            int64_t frameNumber = 0;

            // Read samples in a loop
            while (streamReader.status == AVAssetReaderStatusReading) {
                @autoreleasepool {
                    CMSampleBufferRef sampleBuffer = [trackOutput copyNextSampleBuffer];

                    if (sampleBuffer) {
                        Frame frame = convertSampleBufferToFrame(sampleBuffer, frameNumber++);
                        CFRelease(sampleBuffer);

                        if (!frame.data.empty()) {
                            // Call the callback
                            bool shouldContinue = callback(frame);
                            if (!shouldContinue) {
                                [streamReader cancelReading];
                                break;
                            }
                        }
                    } else {
                        break;
                    }
                }
            }

            [streamReader cancelReading];
        }
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

Frame FrameExtractor::extract_frame(double timestamp) {
    return pImpl->extractFrameAtTime(timestamp);
}

std::vector<Frame> FrameExtractor::extract_frames(const std::vector<double>& timestamps) {
    std::vector<Frame> frames;
    frames.reserve(timestamps.size());

    for (double ts : timestamps) {
        frames.push_back(extract_frame(ts));
    }

    return frames;
}

std::vector<Frame> FrameExtractor::extract_frames_interval(double start, double end, double interval) {
    std::vector<Frame> frames;

    for (double ts = start; ts <= end; ts += interval) {
        frames.push_back(extract_frame(ts));
    }

    return frames;
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

void FrameExtractor::stream_frames(FrameCallback callback) {
    pImpl->streamFrames(0.0, 0.0, callback);
}

void FrameExtractor::stream_frames_from(double start_time, FrameCallback callback) {
    pImpl->streamFrames(start_time, 0.0, callback);
}

void FrameExtractor::stream_frames_range(double start_time, double end_time, FrameCallback callback) {
    pImpl->streamFrames(start_time, end_time, callback);
}

} // namespace videoextractor
