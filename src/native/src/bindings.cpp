#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include "frame_extractor.h"

namespace nb = nanobind;
using namespace viteo;

namespace {

nb::ndarray<nb::c_contig, uint8_t, nb::memview> make_frame_array(
    uint8_t* data, int height, int width, nb::handle owner) {
    return nb::ndarray<nb::c_contig, uint8_t, nb::memview>(
        data,
        {static_cast<size_t>(height), static_cast<size_t>(width), static_cast<size_t>(4)},
        owner);
}

} // namespace

NB_MODULE(_viteo, m) {
    m.doc() = "Hardware-accelerated video frame extraction for Apple Silicon";

    nb::class_<FrameExtractor>(m, "FrameExtractor")
        .def(nb::init<>(), "Create new frame extractor")
        .def("open", &FrameExtractor::open, nb::arg("path"),
            "Open video file for extraction")
        .def("next_frame",
            [](nb::handle self) -> nb::object {
                auto& extractor = nb::cast<FrameExtractor&>(self);
                uint8_t* data;
                {
                    nb::gil_scoped_release release;
                    data = extractor.next_frame();
                }
                if (!data) return nb::none();
                return nb::cast(make_frame_array(data, extractor.height(), extractor.width(), self));
            },
            "Get next frame as a memoryview (None when done)")
        .def("reset", &FrameExtractor::reset, nb::arg("frame_index") = 0,
            "Reset to beginning or specific frame")
        .def_prop_ro("width", &FrameExtractor::width, "Video width")
        .def_prop_ro("height", &FrameExtractor::height, "Video height")
        .def_prop_ro("fps", &FrameExtractor::fps, "Frames per second")
        .def_prop_ro("total_frames", &FrameExtractor::total_frames, "Total frames")
        .def("__iter__", [](nb::object self) { return self; })
        .def("__next__",
            [](nb::handle self) -> nb::ndarray<nb::c_contig, uint8_t, nb::memview> {
                auto& extractor = nb::cast<FrameExtractor&>(self);
                uint8_t* data;
                {
                    nb::gil_scoped_release release;
                    data = extractor.next_frame();
                }
                if (!data) throw nb::stop_iteration();
                return make_frame_array(data, extractor.height(), extractor.width(), self);
            })
        .def("__repr__",
            [](const FrameExtractor& self) {
                return "<FrameExtractor " + std::to_string(self.width()) + "x" +
                       std::to_string(self.height()) + " @ " +
                       std::to_string(self.fps()) + " fps>";
            });
}
