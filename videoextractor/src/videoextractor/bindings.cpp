#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "frame_extractor.h"

namespace nb = nanobind;
using namespace videoextractor;

NB_MODULE(_videoextractor, m) {
    m.doc() = "Hardware-accelerated video frame extraction for Apple Silicon";

    nb::class_<Frame>(m, "Frame")
        .def(nb::init<>())
        .def_ro("width", &Frame::width, "Frame width in pixels")
        .def_ro("height", &Frame::height, "Frame height in pixels")
        .def_ro("timestamp", &Frame::timestamp, "Frame timestamp in seconds")
        .def_ro("frame_number", &Frame::frame_number, "Sequential frame number")
        .def_prop_ro("data", [](const Frame& f) {
            return nb::bytes(reinterpret_cast<const char*>(f.data.data()), f.data.size());
        }, "Raw BGRA frame data as bytes")
        .def("__repr__", [](const Frame& f) {
            return "<Frame #" + std::to_string(f.frame_number) + " " +
                   std::to_string(f.width) + "x" + std::to_string(f.height) +
                   " at " + std::to_string(f.timestamp) + "s (BGRA)>";
        });

    nb::class_<FrameExtractor>(m, "FrameExtractor")
        .def(nb::init<>())
        .def("open", &FrameExtractor::open,
             nb::arg("path"),
             "Open a video file for frame extraction")
        .def("close", &FrameExtractor::close,
             "Close the current video file")
        .def_prop_ro("duration", &FrameExtractor::get_duration,
             "Video duration in seconds")
        .def_prop_ro("width", &FrameExtractor::get_width,
             "Video width in pixels")
        .def_prop_ro("height", &FrameExtractor::get_height,
             "Video height in pixels")
        .def_prop_ro("fps", &FrameExtractor::get_fps,
             "Video frames per second")
        .def("start_streaming", &FrameExtractor::start_streaming,
             nb::arg("start_time") = 0.0,
             nb::arg("end_time") = 0.0,
             "Start a streaming session for batch frame access")
        .def("next_frames_batch", [](FrameExtractor& self, size_t max_frames) {
            std::vector<Frame> frames;
            size_t count;

            // Release GIL during frame decoding for better parallelism
            {
                nb::gil_scoped_release release;
                count = self.next_frames_batch(frames, max_frames);
            }

            return frames;
        }, nb::arg("max_frames") = 32,
        "Get next batch of frames (returns list of Frame objects)")
        .def("is_streaming", &FrameExtractor::is_streaming,
             "Check if a streaming session is currently active")
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
