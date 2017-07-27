import numpy as np

import cts.model as model
import cpp_cts

def L_shaped_context(image, y, x):
    """This grabs the L-shaped context around a given pixel.

    Out-of-bounds values are set to 0xFFFFFFFF."""
    context = [0xFFFFFFFF] * 4
    if x > 0:
        context[3] = image[y][x - 1]

    if y > 0:
        context[2] = image[y - 1][x]
        context[1] = image[y - 1][x - 1] if x > 0 else 0
        context[0] = image[y - 1][x + 1] if x < image.shape[1] - 1 else 0

    # The most important context symbol, 'left', comes last.
    return context

def L_shaped_context_binary(image, y, x):
    """This grabs the L-shaped context around a given pixel.

    Out-of-bounds values are set to 0xFFFFFFFF."""
    context = [0xFFFFFFFF] * 4
    if x > 0:
        context[3] = image[y][x - 1]

    if y > 0:
        context[2] = image[y - 1][x]
        context[1] = image[y - 1][x - 1] if x > 0 else 0
        context[0] = image[y - 1][x + 1] if x < image.shape[1] - 1 else 0

    # The most important context symbol, 'left', comes last.
    return np.transpose(np.reshape(np.unpackbits(np.array(context, dtype=np.uint8)), (-1, 8)))

class LocationDependentDensityModel(object):
    """A density model for Freeway frames.

    This is exactly the same as the ConvolutionalDensityModel, except that we use one model for each
    pixel location.
    """

    def __init__(self, frame_shape, to_symbol_func, encoding_length, context_func, alphabet=None):

        # For efficiency, we'll pre-process the frame into our internal representation.
        self.symbol_frame = np.zeros((frame_shape[0:2]), dtype=np.uint32)

        context_length = len(context_func(self.symbol_frame, -1, -1))
        # self.models = np.zeros(frame_shape[0:2], dtype=object)
        self.cpp_models = np.zeros(frame_shape[0:2], dtype=object)

        for y in range(frame_shape[0]):
            for x in range(frame_shape[1]):
                # self.models[y, x] = model.CTS(context_length=context_length, alphabet=alphabet)
                # self.cpp_models[y, x] = cpp_cts.CPP_CTS(context_length, encoding_length)
                self.cpp_models[y, x] = cpp_cts.CPP_CTS(4, 3)

        self.context_func = context_func
        self.to_symbol_func = to_symbol_func

    def prob_update(self, frame):
        self.symbol_frame = self.to_symbol_func(frame, self.symbol_frame)

        total_p = 1.0
        total_p_prime = 1.0
        for y in range(self.symbol_frame.shape[0]):
            for x in range(self.symbol_frame.shape[1]):
                # context = self.context_func(self.symbol_frame, y, x)
                # colour = self.symbol_frame[y, x]
                # total_log_probability += self.models[y, x].update(context=context, symbol=colour)

                cts = self.cpp_models[y, x]
                context = L_shaped_context_binary(self.symbol_frame, y, x)
                colour = self.symbol_frame[y, x]
                p, p_prime = cts.prob_update(colour, context)
                total_p *= p
                total_p_prime *= p_prime

        return total_p, total_p_prime

    def update(self, frame):
        self.symbol_frame = self.to_symbol_func(frame, self.symbol_frame)

        total_probability = 1.0
        for y in range(self.symbol_frame.shape[0]):
            for x in range(self.symbol_frame.shape[1]):
                # context = self.context_func(self.symbol_frame, y, x)
                # colour = self.symbol_frame[y, x]
                # total_log_probability += self.models[y, x].update(context=context, symbol=colour)

                cts = self.cpp_models[y, x]
                context = L_shaped_context_binary(self.symbol_frame, y, x)
                colour = self.symbol_frame[y, x]
                total_probability *= cts.update(colour, context)

        return total_probability

    def prob(self, frame):
        self.symbol_frame = self.to_symbol_func(frame, self.symbol_frame)

        total_probability = 1.0
        for y in range(self.symbol_frame.shape[0]):
            for x in range(self.symbol_frame.shape[1]):
                # context = self.context_func(self.symbol_frame, y, x)
                # colour = self.symbol_frame[y, x]
                # total_log_probability += self.models[y, x].log_prob(context=context, symbol=colour)

                cts = self.cpp_models[y, x]
                context = L_shaped_context_binary(self.symbol_frame, y, x)
                colour = self.symbol_frame[y, x]
                total_probability *= cts.update(colour, context)

        return total_probability
