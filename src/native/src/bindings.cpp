#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "frame_extractor.h"
#include <stdexcept>
#include <Python.h>

namespace nb = nanobind;
using namespace viteo;

class FrameIterator {
private:
    FrameExtractor* extractor;
    nb::object buffer_obj;
    uint8_t* buffer_ptr;
    size_t batch_size;
    size_t current_batch_size;
    size_t current_index;
    int width, height;
    nb::object mlx_module;
    nb::object mx_array;

public:
    FrameIterator(FrameExtractor* ext, nb::object buf, size_t batch)
        : extractor(ext), buffer_obj(buf), batch_size(batch),
          current_batch_size(0), current_index(0) {

        width = extractor->width();
        height = extractor->height();

        mlx_module = nb::module_::import_("mlx.core");
        mx_array = mlx_module.attr("array");

        Py_buffer view;
        if (PyObject_GetBuffer(buffer_obj.ptr(), &view, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) != 0) {
            throw std::runtime_error("Failed to get buffer from MLX array");
        }
        buffer_ptr = static_cast<uint8_t*>(view.buf);
        PyBuffer_Release(&view);

        load_next_batch();
    }

    void load_next_batch() {
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

        size_t frame_size = width * height * 4;
        size_t offset = current_index * frame_size;
        nb::object py_memview = nb::steal(PyMemoryView_FromMemory(
            reinterpret_cast<char*>(buffer_ptr + offset),
            frame_size, PyBUF_READ
        ));

        nb::object mx_uint8 = mlx_module.attr("uint8");
        nb::object frame = mx_array(py_memview, mx_uint8);

        nb::object shape = nb::make_tuple(height, width, 4);
        frame = frame.attr("reshape")(shape);

        current_index++;
        return frame;
    }
};

NB_MODULE(_viteo, m) {
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
        .def("get_next_batch",
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
        .def("__repr__",
            [](const FrameExtractor& self) {
                return "<FrameExtractor " + std::to_string(self.width()) + "x" +
                       std::to_string(self.height()) + " @ " +
                       std::to_string(self.fps()) + " fps>";
            });

    nb::class_<FrameIterator>(m, "FrameIterator")
        .def(nb::init<FrameExtractor*, nb::object, size_t>(),
            nb::arg("extractor"), nb::arg("buffer"), nb::arg("batch_size"),
            "Create iterator for frame extraction")
        .def("__next__", &FrameIterator::next)
        .def("__iter__", [](nb::object self) { return self; });
}