#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>
#include "frame_extractor.h"

namespace nb = nanobind;
using namespace videoextractor;

NB_MODULE(_videoextractor, m) {
    m.doc() = "Hardware-accelerated video frame extraction for Apple Silicon";

    // Frame class
    nb::class_<Frame>(m, "Frame")
        .def(nb::init<>())
        .def_ro("width", &Frame::width, "Frame width in pixels")
        .def_ro("height", &Frame::height, "Frame height in pixels")
        .def_ro("timestamp", &Frame::timestamp, "Frame timestamp in seconds")
        .def_ro("frame_number", &Frame::frame_number, "Sequential frame number")
        .def_prop_ro("data", [](const Frame& f) {
            return nb::bytes(reinterpret_cast<const char*>(f.data.data()), f.data.size());
        }, "Raw RGB frame data as bytes")
        .def("to_numpy", [](const Frame& f) {
            // Return shape info for numpy array creation in Python
            return nb::make_tuple(f.height, f.width, 3);
        }, "Get shape information for numpy array conversion")
        .def("__repr__", [](const Frame& f) {
            return "<Frame #" + std::to_string(f.frame_number) + " " +
                   std::to_string(f.width) + "x" + std::to_string(f.height) +
                   " at " + std::to_string(f.timestamp) + "s>";
        });

    // FrameExtractor class
    nb::class_<FrameExtractor>(m, "FrameExtractor")
        .def(nb::init<>())
        .def("open", &FrameExtractor::open,
             nb::arg("path"),
             "Open a video file for frame extraction")
        .def("close", &FrameExtractor::close,
             "Close the current video file")
        .def("extract_frame", &FrameExtractor::extract_frame,
             nb::arg("timestamp"),
             "Extract a single frame at the given timestamp (in seconds)")
        .def("extract_frames", &FrameExtractor::extract_frames,
             nb::arg("timestamps"),
             "Extract frames at specified timestamps")
        .def("extract_frames_interval", &FrameExtractor::extract_frames_interval,
             nb::arg("start"),
             nb::arg("end"),
             nb::arg("interval"),
             "Extract frames at regular intervals between start and end time")
        .def_prop_ro("duration", &FrameExtractor::get_duration,
             "Video duration in seconds")
        .def_prop_ro("width", &FrameExtractor::get_width,
             "Video width in pixels")
        .def_prop_ro("height", &FrameExtractor::get_height,
             "Video height in pixels")
        .def_prop_ro("fps", &FrameExtractor::get_fps,
             "Video frames per second")
        .def("stream_frames", [](FrameExtractor& self, nb::object callback) {
            // Wrapper to handle Python callbacks with GIL management
            self.stream_frames([callback](const Frame& frame) -> bool {
                nb::gil_scoped_acquire acquire;
                try {
                    nb::object result = callback(frame);
                    // If callback returns None or True, continue streaming
                    if (result.is_none()) {
                        return true;
                    }
                    return nb::cast<bool>(result);
                } catch (const std::exception& e) {
                    // If callback raises exception, stop streaming
                    return false;
                }
            });
        }, nb::arg("callback"),
        nb::call_guard<nb::gil_scoped_release>(),
        "Stream all frames from the beginning. Callback receives each frame and returns True to continue or False to stop.")
        .def("stream_frames_from", [](FrameExtractor& self, double start_time, nb::object callback) {
            self.stream_frames_from(start_time, [callback](const Frame& frame) -> bool {
                nb::gil_scoped_acquire acquire;
                try {
                    nb::object result = callback(frame);
                    if (result.is_none()) {
                        return true;
                    }
                    return nb::cast<bool>(result);
                } catch (const std::exception& e) {
                    return false;
                }
            });
        }, nb::arg("start_time"), nb::arg("callback"),
        nb::call_guard<nb::gil_scoped_release>(),
        "Stream frames starting from a specific timestamp. Callback receives each frame and returns True to continue or False to stop.")
        .def("stream_frames_range", [](FrameExtractor& self, double start_time, double end_time, nb::object callback) {
            self.stream_frames_range(start_time, end_time, [callback](const Frame& frame) -> bool {
                nb::gil_scoped_acquire acquire;
                try {
                    nb::object result = callback(frame);
                    if (result.is_none()) {
                        return true;
                    }
                    return nb::cast<bool>(result);
                } catch (const std::exception& e) {
                    return false;
                }
            });
        }, nb::arg("start_time"), nb::arg("end_time"), nb::arg("callback"),
        nb::call_guard<nb::gil_scoped_release>(),
        "Stream frames between start and end timestamps. Callback receives each frame and returns True to continue or False to stop.")
        .def("__enter__", [](FrameExtractor& self) -> FrameExtractor& {
            return self;
        })
        .def("__exit__", [](FrameExtractor& self, nb::object, nb::object, nb::object) {
            self.close();
        })
        .def("__repr__", [](const FrameExtractor& fe) {
            return "<FrameExtractor " + std::to_string(fe.get_width()) + "x" +
                   std::to_string(fe.get_height()) + " @ " +
                   std::to_string(fe.get_fps()) + " fps>";
        });
}
