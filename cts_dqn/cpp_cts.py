import ctypes
from ctypes import cdll
import numpy as np
from numpy.ctypeslib import ndpointer

lib = cdll.LoadLibrary('/home/maroderi/projects/context_weighting_tree/libcwt.so')
constructCTS = lib.construct_symbolic_CTS
constructCTS.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
constructCTS.restype = ctypes.c_void_p
psuedo_count_for_image = lib.psuedo_count_for_image_symbolic
psuedo_count_for_image.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_uint8)]
psuedo_count_for_image.restype = ctypes.c_double


class CPP_CTS():
    def __init__(self, width, height, alphabet_size):
        self.ptr = constructCTS(width, height, alphabet_size)

    def psuedo_count_for_image(self, image):
        return psuedo_count_for_image(self.ptr, np.reshape(image, (-1,)))
