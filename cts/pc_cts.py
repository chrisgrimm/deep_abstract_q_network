import numpy as np

import cts.model as model

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

class LocationDependentDensityModel(object):
    """A density model for Freeway frames.

    This is exactly the same as the ConvolutionalDensityModel, except that we use one model for each
    pixel location.
    """

    def __init__(self, frame_shape, to_symbol_func, context_func, alphabet=None):
        self.models = np.zeros(frame_shape[0:2], dtype=object)

        for y in range(frame_shape[0]):
            for x in range(frame_shape[1]):
                self.models[y, x] = model.CTS(context_length=4, alphabet=alphabet)

        self.context_func = context_func
        self.to_symbol_func = to_symbol_func

    def update(self, frame):
        total_log_probability = 0.0
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                context = self.context_func(frame, y, x)
                colour = frame[y, x]
                total_log_probability += self.models[y, x].update(context=context, symbol=colour)

        return total_log_probability

    def log_prob(self, frame):
        total_log_probability = 0.0
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                context = self.context_func(frame, y, x)
                colour = frame[y, x]
                total_log_probability += self.models[y, x].log_prob(context=context, symbol=colour)

        return total_log_probability

    def psuedo_count_for_image(self, frame):
        p = self.update(frame)
        rec_p = self.log_prob(frame)

        p_exp = np.exp(p)
        rec_p_exp = np.exp(rec_p)

        if (p_exp >= rec_p_exp):
            return 0
        else:
            return p_exp * (1 - rec_p_exp)/(rec_p_exp - p_exp)
