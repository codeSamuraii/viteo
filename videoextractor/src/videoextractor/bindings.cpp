
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "frame_extractor.h"
#include <stdexcept>

#include <Python.h>

namespace nb = nanobind;
using namespace videoextractor;

NB_MODULE(_videoextractor, m) {
    m.doc() = "Hardware-accelerated video frame streaming for Apple Silicon (MLX-optimized)";

    nb::class_<FrameExtractor>(m, "FrameExtractor")
       .def(nb::init<>())
       .def("open", &FrameExtractor::open,
           nb::arg("path"),
           "Open a video file for frame extraction")
       .def("close", &FrameExtractor::close,
           "Close the current video file")
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
       .def("next_frames_batch_to_buffer",
          [](FrameExtractor& self, nb::handle py_buffer, size_t max_frames) {
             // Use Python C API to get buffer info
             Py_buffer view;
             if (PyObject_GetBuffer(py_buffer.ptr(), &view, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) != 0) {
                throw std::runtime_error("Object does not support writable, C-contiguous buffer interface");
             }
             if (view.ndim != 4 || view.shape[3] != 4 || view.itemsize != 1) {
                PyBuffer_Release(&view);
                throw std::runtime_error("Buffer must be (batch, height, width, 4), dtype=uint8");
             }
             int width = static_cast<int>(view.shape[2]);
             int height = static_cast<int>(view.shape[1]);
             size_t stride = width * height * 4;
             uint8_t* out_ptr = static_cast<uint8_t*>(view.buf);
             size_t frames_written = 0;
             {
                nb::gil_scoped_release release;
                frames_written = self.next_frames_batch_to_buffer(out_ptr, max_frames, width, height, stride);
             }
             PyBuffer_Release(&view);
             return frames_written;
          },
          nb::arg("buffer"), nb::arg("max_frames") = 32,
          "Decode up to max_frames directly into a (batch, height, width, 4) MLX or numpy array (dtype=uint8). Returns number of frames written.")
       .def("is_streaming", &FrameExtractor::is_streaming,
           "Check if a streaming session is currently active")
       .def("__enter__", [](FrameExtractor& self) -> FrameExtractor& { return self; })
       .def("__exit__", [](FrameExtractor& self, nb::object, nb::object, nb::object) { self.close(); })
       .def("__repr__", [](const FrameExtractor& fe) {
          return "<FrameExtractor " + std::to_string(fe.get_width()) + "x" +
                std::to_string(fe.get_height()) + " @ " +
                std::to_string(fe.get_fps()) + " fps>";
       });
}
