#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include "frame_extractor.h"
#include <memory>
#include <vector>

namespace nb = nanobind;
using namespace viteo;

namespace {

nb::ndarray<nb::c_contig, uint8_t, nb::memview> make_frame_array(
    const std::shared_ptr<std::vector<uint8_t>>& buffer,
    int height,
    int width) {
    // Keep buffer alive as long as Python holds the view
    auto* holder = new std::shared_ptr<std::vector<uint8_t>>(buffer);
    nb::capsule owner(holder, [](void* p) noexcept {
        delete static_cast<std::shared_ptr<std::vector<uint8_t>>*>(p);
    });
    return nb::ndarray<nb::c_contig, uint8_t, nb::memview>(
        (*holder)->data(),
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
            [](FrameExtractor& self) -> nb::object {
                std::shared_ptr<std::vector<uint8_t>> frame_buffer;
                {
                    nb::gil_scoped_release release;
                    frame_buffer = self.next_frame();
                }
                if (!frame_buffer) return nb::none();
                return nb::cast(make_frame_array(frame_buffer, self.height(), self.width()));
            },
            "Get next frame as a read-only ndarray (None when done)")
        .def("reset", &FrameExtractor::reset, nb::arg("frame_index") = 0,
            "Reset to beginning or specific frame")
        .def_prop_ro("width", &FrameExtractor::width, "Video width")
        .def_prop_ro("height", &FrameExtractor::height, "Video height")
        .def_prop_ro("fps", &FrameExtractor::fps, "Frames per second")
        .def_prop_ro("total_frames", &FrameExtractor::total_frames, "Total frames")
        .def("__iter__", [](nb::object self) { return self; })
        .def("__next__",
            [](FrameExtractor& self) -> nb::ndarray<nb::c_contig, uint8_t, nb::memview> {
                std::shared_ptr<std::vector<uint8_t>> frame_buffer;
                {
                    nb::gil_scoped_release release;
                    frame_buffer = self.next_frame();
                }
                if (!frame_buffer) throw nb::stop_iteration();
                return make_frame_array(frame_buffer, self.height(), self.width());
            })
        .def("__repr__",
            [](const FrameExtractor& self) {
                return "<FrameExtractor " + std::to_string(self.width()) + "x" +
                       std::to_string(self.height()) + " @ " +
                       std::to_string(self.fps()) + " fps>";
            });
}
