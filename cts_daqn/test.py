import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc
import sys
import time

import cts.model as model

def rgb_to_symbol(channels):
    """Converts an RGB triple into an atomic colour (24-bit integer).
    """
    return (channels[0] << 16) | (channels[1] << 8) | (channels[2] << 0)


def symbol_to_rgb(colour):
    """Inverse operation of rgb_to_symbol.

    Returns: a (r, g, b) tuple.
    """
    return (colour >> 16, (colour >> 8) & 0xFF, (colour >> 0) & 0xFF)


def rgb_to_symbols(frame, output):
    """Preprocesses the given frame into a CTS-compatible representation.
    """
    assert (frame.shape[:2] == output.shape)
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            output[y, x] = rgb_to_symbol(frame[y, x])

    return output


def symbols_to_rgb(frame, output):
    """Inverse of rgb_to_symbols.
    """
    assert (frame.shape == output.shape[:2])
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            output[y, x] = symbol_to_rgb(frame[y, x])

    return output


def freeway_generator(start_frame=0, max_frames=-1):
    """Generator of Freeway frames.

    Args:
        start_frame: The starting frame. Frames prior to this one are discarded.
        max_frames: Maximum number of frames to return. If -1, return all frames.
    """
    frame_counter = 0
    if max_frames >= 0:
        end_frame = start_frame + max_frames

    for filename in glob.glob('freeway-frames/*.png'):
        if frame_counter < start_frame:
            frame_counter += 1
            continue
        elif max_frames >= 0 and frame_counter >= end_frame:
            return
        else:
            frame_counter += 1
            yield misc.imread(filename)


class ConvolutionalMarginalDensityModel(object):
    """A density model for Freeway frames."""

    def __init__(self, frame_shape):
        """Constructor.

        Args:
            init_frame: A sample frame (numpy array) from which we determine the shape and type of our data.
        """
        self.convolutional_model = model.CTS(context_length=0)
        self.frame_shape = frame_shape

    def update(self, frame):
        assert (frame.shape == self.frame_shape)

        total_log_probability = 0.0
        # We simply apply the CTS update to each pixel.
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                # Convert all 3 channels to an atomic colour.
                colour = rgb_to_symbol(frame[y, x])
                total_log_probability += self.convolutional_model.update(context=[], symbol=colour)

        return total_log_probability

    def sample(self):
        output_frame = np.zeros(self.frame_shape, dtype=np.uint8)

        for y in range(output_frame.shape[0]):
            for x in range(output_frame.shape[1]):
                # Use rejection sampling to avoid generating non-Atari colours.
                colour = self.convolutional_model.sample(context=[], rejection_sampling=True)
                output_frame[y, x] = symbol_to_rgb(colour)

        return output_frame


class LocationDependentDensityModel(object):
    """A density model for Freeway frames.

    This is exactly the same as the ConvolutionalDensityModel, except that we use one model for each
    pixel location.
    """

    def __init__(self, frame_shape, context_functor, alphabet=None):
        """Constructor.

        Args:
            init_frame: A sample frame (numpy array) from which we determine the shape and type of our data.
            context_functor: Function mapping image x position to a context.
        """
        # For efficiency, we'll pre-process the frame into our internal representation.
        self.symbol_frame = np.zeros((frame_shape[0:2]), dtype=np.uint32)

        context_length = len(context_functor(self.symbol_frame, -1, -1))
        self.models = np.zeros(frame_shape[0:2], dtype=object)

        for y in range(frame_shape[0]):
            for x in range(frame_shape[1]):
                self.models[y, x] = model.CTS(context_length=context_length, alphabet=alphabet)

        self.context_functor = context_functor

    def update(self, frame):
        rgb_to_symbols(frame, self.symbol_frame)

        total_log_probability = 0.0
        for y in range(self.symbol_frame.shape[0]):
            for x in range(self.symbol_frame.shape[1]):
                context = self.context_functor(self.symbol_frame, y, x)
                colour = self.symbol_frame[y, x]
                total_log_probability += self.models[y, x].update(context=context, symbol=colour)

        return total_log_probability


    def sample(self):
        output_frame = np.zeros(self.symbol_frame.shape + (3,), dtype=np.uint8)
        print (output_frame.shape)

        for y in range(self.symbol_frame.shape[0]):
            for x in range(self.symbol_frame.shape[1]):
                # From a programmer's perspective, this is why we must respect the chain rule: otherwise
                # we condition on garbage.
                context = self.context_functor(self.symbol_frame, y, x)
                self.symbol_frame[y, x] = self.models[y, x].sample(context=context, rejection_sampling=True)

        return symbols_to_rgb(self.symbol_frame, output_frame)

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

def train_freeway_density_model(model, num_frames):
    start_time = time.time()
    total_log_probability = 0.0

    sys.stdout.write('Training on {} frames '.format(num_frames))
    for frame in freeway_generator(start_frame=0, max_frames=num_frames):
        total_log_probability += model.update(frame)
        sys.stdout.write('.')
    sys.stdout.write('\n')

    print ('Training time: {:.1f} seconds'.format(time.time() - start_time))
    print ('Loss (in bytes per frame): {:.2f}'.format(-total_log_probability / math.log(2) / 8 / num_frames))














FRAME_SHAPE = next(freeway_generator(0, 1)).shape
print ('Frame shape is {}'.format(FRAME_SHAPE))

location_dependent_model = LocationDependentDensityModel(frame_shape=(210, 320, 3), context_functor=L_shaped_context)
train_freeway_density_model(location_dependent_model, num_frames=20)

sampled_frame = location_dependent_model.sample()

plt.imshow(sampled_frame)
plt.show()