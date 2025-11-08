#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "frame_extractor.h"
#include <Python.h>

namespace nb = nanobind;
using namespace viteo;

/// Create MLX array from raw BGRA buffer
mlx::core::array create_mlx_array(uint8_t* data, int height, int width) {
    auto arr = mlx::core::array(
        data,
        mlx::core::Shape{ (int32_t)height, (int32_t)width, int32_t(4) },
        mlx::core::uint8
    );

    // Eval the array
    mlx::core::eval({arr});

    return arr;
}

NB_MODULE(_viteo, m) {
    m.doc() = "Hardware-accelerated video frame extraction for Apple Silicon";

    nb::class_<FrameExtractor>(m, "FrameExtractor")
        .def(nb::init<>(), "Create new frame extractor")
        .def("open", &FrameExtractor::open, nb::arg("path"),
            "Open video file for extraction")
        .def("next_frame",
            [](FrameExtractor& self) -> mlx::core::array {
                uint8_t* frame_data;
                {
                    nb::gil_scoped_release release;
                    frame_data = self.next_frame();
                }
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
            [](FrameExtractor& self) -> mlx::core::array {
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
