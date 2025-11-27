#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>

#include "mlx/array.h"
#include "frame_extractor.h"

namespace nb = nanobind;

using namespace viteo;

// Helper to convert mlx::core::array to Python mlx.core.array
nb::object to_python_array(const mlx::core::array& arr) {
    if (arr.size() == 0) return nb::none();

    // Evaluate to ensure data is ready
    mlx::core::eval({arr});

    // Get raw pointer and shape info
    auto shape = arr.shape();
    int64_t h = shape[0];
    int64_t w = shape[1];
    int64_t c = shape[2];

    // Create numpy array view of the data
    const uint8_t* data = arr.data<uint8_t>();
    size_t np_shape[3] = {(size_t)h, (size_t)w, (size_t)c};

    auto np_arr = nb::ndarray<nb::numpy, const uint8_t>(
        (void*)data, 3, np_shape
    );

    // Convert to MLX array via Python
    nb::module_ mx = nb::module_::import_("mlx.core");
    return mx.attr("array")(np_arr);
}

NB_MODULE(_viteo, m) {
    m.doc() = "Hardware-accelerated video frame extraction for Apple Silicon";

    nb::class_<FrameExtractor>(m, "FrameExtractor")
        .def(nb::init<>(), "Create new frame extractor")
        .def("open", &FrameExtractor::open, nb::arg("path"),
            "Open video file for extraction")
        .def("next_frame",
            [](FrameExtractor& self) -> nb::object {
                mlx::core::array frame({}, mlx::core::uint8);
                {
                    nb::gil_scoped_release release;
                    frame = self.next_frame();
                }
                return to_python_array(frame);
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
                mlx::core::array frame({}, mlx::core::uint8);
                {
                    nb::gil_scoped_release release;
                    frame = self.next_frame();
                }
                if (frame.size() == 0) throw nb::stop_iteration();
                return to_python_array(frame);
            })
        .def("__repr__",
            [](const FrameExtractor& self) {
                return "<FrameExtractor " + std::to_string(self.width()) + "x" +
                       std::to_string(self.height()) + " @ " +
                       std::to_string(self.fps()) + " fps>";
            });
}
