#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "frame_extractor.h"
#include <stdexcept>
#include <Python.h>

namespace nb = nanobind;
using namespace videoextractor;

// Custom iterator class for frame extraction
class FrameIterator {
private:
    FrameExtractor* extractor;
    nb::object buffer_obj;  // Keep MLX buffer alive
    uint8_t* buffer_ptr;
    size_t batch_size;
    size_t current_batch_size;
    size_t current_index;
    int width, height;

public:
    FrameIterator(FrameExtractor* ext, nb::object buf, size_t batch)
        : extractor(ext), buffer_obj(buf), batch_size(batch),
          current_batch_size(0), current_index(0) {

        width = extractor->width();
        height = extractor->height();

        // Get buffer pointer from MLX array
        Py_buffer view;
        if (PyObject_GetBuffer(buffer_obj.ptr(), &view, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) != 0) {
            throw std::runtime_error("Failed to get buffer from MLX array");
        }
        buffer_ptr = static_cast<uint8_t*>(view.buf);
        PyBuffer_Release(&view);

        // Load first batch
        load_next_batch();
    }

    void load_next_batch() {
        // Release GIL for performance during extraction
        nb::gil_scoped_release release;
        current_batch_size = extractor->extract_batch(buffer_ptr, batch_size);
        current_index = 0;
    }

    nb::object next() {
        if (current_index >= current_batch_size) {
            if (current_batch_size == 0) {
                throw nb::stop_iteration();
            }
            load_next_batch();
            if (current_batch_size == 0) {
                throw nb::stop_iteration();
            }
        }

        // Return a view of the current frame in the buffer
        // Import MLX dynamically to create the view
        nb::module_ mlx = nb::module_::import_("mlx.core");
        nb::object mx_array_type = mlx.attr("array");

        // Calculate offset for current frame
        size_t frame_size = width * height * 4;
        size_t offset = current_index * frame_size;

        // Create MLX array view from buffer slice
        nb::object py_memview = nb::steal(PyMemoryView_FromMemory(
            reinterpret_cast<char*>(buffer_ptr + offset),
            frame_size, PyBUF_READ));

        // Reshape to (height, width, 4) and convert to MLX array
        nb::object np = nb::module_::import_("numpy");
        nb::object frombuffer = np.attr("frombuffer");
        nb::object np_array = frombuffer(py_memview, "uint8");
        np_array = np_array.attr("reshape")(height, width, 4);

        nb::object frame = mx_array_type(np_array);
        current_index++;

        return frame;
    }
};

NB_MODULE(_videoextractor, m) {
    m.doc() = "Hardware-accelerated video frame extraction for Apple Silicon with MLX";

    nb::class_<FrameExtractor>(m, "FrameExtractor")
        .def(nb::init<>())
        .def("open", &FrameExtractor::open,
            nb::arg("path"),
            "Open a video file")
        .def_prop_ro("width", &FrameExtractor::width,
            "Video width in pixels")
        .def_prop_ro("height", &FrameExtractor::height,
            "Video height in pixels")
        .def_prop_ro("fps", &FrameExtractor::fps,
            "Video frames per second")
        .def_prop_ro("total_frames", &FrameExtractor::total_frames,
            "Estimated total number of frames")
        .def("reset", &FrameExtractor::reset,
            nb::arg("frame_index") = 0,
            "Reset to beginning or specific frame")
        .def("extract_batch_raw",
            [](FrameExtractor& self, nb::handle buffer, size_t batch_size) {
                Py_buffer view;
                if (PyObject_GetBuffer(buffer.ptr(), &view, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) != 0) {
                    throw std::runtime_error("Buffer must be writable and C-contiguous");
                }

                uint8_t* ptr = static_cast<uint8_t*>(view.buf);
                size_t frames_extracted;
                {
                    nb::gil_scoped_release release;
                    frames_extracted = self.extract_batch(ptr, batch_size);
                }

                PyBuffer_Release(&view);
                return frames_extracted;
            },
            nb::arg("buffer"), nb::arg("batch_size"),
            "Extract frames directly into MLX buffer (low-level)")
        .def("__iter__",
            [](nb::object self) {
                // Create iterator with internal buffer
                nb::object ext_obj = self;
                FrameExtractor& ext = nb::cast<FrameExtractor&>(self);

                // Import MLX and create buffer
                nb::module_ mlx = nb::module_::import_("mlx.core");
                nb::object mx_zeros = mlx.attr("zeros");
                nb::object mx_uint8 = mlx.attr("uint8");

                // Create buffer for batch processing (default 32 frames)
                size_t batch_size = 32;
                nb::object buffer = mx_zeros(
                    nb::make_tuple(batch_size, ext.height(), ext.width(), 4),
                    mx_uint8);

                // Create and return iterator
                return nb::cast(new FrameIterator(&ext, buffer, batch_size),
                    nb::rv_policy::take_ownership);
            },
            "Return iterator over frames")
        .def("__repr__",
            [](const FrameExtractor& self) {
                return "<FrameExtractor " + std::to_string(self.width()) + "x" +
                       std::to_string(self.height()) + " @ " +
                       std::to_string(self.fps()) + " fps>";
            });

    nb::class_<FrameIterator>(m, "FrameIterator")
        .def("__next__", &FrameIterator::next)
        .def("__iter__", [](nb::object self) { return self; });
}