import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy

class CUDAAccelerator:
    def __init__(self):
        self.mod = None

    def compile(self, code):
        self.mod = SourceModule(code)

    def get_function(self, name):
        if self.mod is None:
            raise ValueError("No CUDA code compiled")
        return self.mod.get_function(name)

    def allocate(self, size):
        return cuda.mem_alloc(size)

    def to_device(self, host_array):
        device_array = cuda.mem_alloc(host_array.nbytes)
        cuda.memcpy_htod(device_array, host_array)
        return device_array

    def from_device(self, device_array, shape):
        host_array = numpy.empty(shape, dtype = numpy.float32)
        cuda.memcpy_dtoh(host_array, device_array)
        return host_array
