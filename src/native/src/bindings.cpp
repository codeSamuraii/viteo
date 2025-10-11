#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "frame_extractor.h"
#include <Python.h>

namespace nb = nanobind;
using namespace viteo;

/// Create MLX array from raw BGRA buffer
nb::object create_mlx_array(uint8_t* data, int height, int width) {
    if (!data) return nb::none();

    // Import MLX
    nb::object mlx = nb::module_::import_("mlx.core");
    nb::object mx_array = mlx.attr("array");
    nb::object mx_uint8 = mlx.attr("uint8");

    // Create memory view
    size_t size = height * width * 4;
    nb::object memview = nb::steal(PyMemoryView_FromMemory(
        reinterpret_cast<char*>(data), size, PyBUF_READ
    ));

    // Create MLX array and reshape
    nb::object arr = mx_array(memview, mx_uint8);
    return arr.attr("reshape")(nb::make_tuple(height, width, 4));
}

NB_MODULE(_viteo, m) {
    m.doc() = "Hardware-accelerated video frame extraction for Apple Silicon";

    nb::class_<FrameExtractor>(m, "FrameExtractor")
        .def(nb::init<size_t>(), nb::arg("batch_size") = 8, "Create new frame extractor")
        .def("open", &FrameExtractor::open, nb::arg("path"),
            "Open video file for extraction")
        .def("next_frame",
            [](FrameExtractor& self) -> nb::object {
                uint8_t* frame_data;
                {
                    nb::gil_scoped_release release;
                    frame_data = self.next_frame();
                }
                if (!frame_data) return nb::none();
                return create_mlx_array(frame_data, self.height(), self.width());
            },
            "Get next frame as MLX array (None when done)")
        .def("reset", &FrameExtractor::reset, nb::arg("frame_index") = 0,
            "Reset to beginning or specific frame")
        .def_prop_ro("width", &FrameExtractor::width, "Video width")
        .def_prop_ro("height", &FrameExtractor::height, "Video height")
        .def_prop_ro("fps", &FrameExtractor::fps, "Frames per second")
        .def_prop_ro("total_frames", &FrameExtractor::total_frames, "Total frames")
        .def("__iter__", [](nb::object self) { return self; })
        .def("__next__",
            [](FrameExtractor& self) -> nb::object {
                uint8_t* frame_data;
                {
                    nb::gil_scoped_release release;
                    frame_data = self.next_frame();
                }
                if (!frame_data) throw nb::stop_iteration();
                return create_mlx_array(frame_data, self.height(), self.width());
            })
        .def("__repr__",
            [](const FrameExtractor& self) {
                return "<FrameExtractor " + std::to_string(self.width()) + "x" +
                       std::to_string(self.height()) + " @ " +
                       std::to_string(self.fps()) + " fps>";
            });
}
