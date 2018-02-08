import parse_results
import matplotlib.pyplot as plt
import numpy as np


# Paper params
# figsize=(4, 3)

# Talk params
figsize=(10, 7.5)


# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

tableau10 = []
tableau5 = []
for i, c in enumerate(tableau20):
    if i % 2 == 0:
        tableau10.append(c)
    if i % 4 == 0:
        tableau5.append(c)


# This function takes an array of numbers and smoothes them out.
# Smoothing is useful for making plots a little easier to read.
def sliding_mean(data_array, window=5):
    data_array = np.array(data_array)
    new_list = []
    for i in range(len(data_array)):
        indices = range(max(i - window + 1, 0),
                        min(i + window + 1, len(data_array)))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)

    return np.array(new_list)


def brighten_color(r, g, b):
    return r/2 + 0.5, g/2 + 0.5, b/2 + 0.5


def plot_data(
        x, ys, title, xlabel, ylabel, labels=None,
        error_tops=None, error_bots=None,
        ylim=None, yticks=None,
        save_file=None, legend_loc='best',
        colors=None, dashes=None
):
    if colors is None:
        if len(ys) <= 10:
            colors = tableau10
        else:
            colors = tableau20

    # You typically want your plot to be ~1.33x wider than tall.
    # Common sizes: (10, 7.5) and (12, 9)
    plt.figure(figsize=(4, 3))

    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    # plt.xticks(range(1850, 2011, 20), fontsize=14)
    if yticks is not None:
        plt.yticks(yticks, fontsize=16)
    else:
        plt.yticks(fontsize=16)

    plt.xticks(fontsize=16)

    # Along the same vein, make sure your axis labels are large
    # enough to be easily read as well. Make them slightly larger
    # than your axis tick labels so they stand out.
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)

    # Use matplotlib's fill_between() call to create error bars.
    # Use the dark blue "#3F5D7D" as a nice fill color.
    if error_bots is not None and error_tops is not None:
        for i, (error_bot, error_top) in enumerate(zip(error_bots, error_tops)):
            plt.fill_between(x, error_bot, error_top, color=brighten_color(*colors[i]))

    # Plot the means as a white line in between the error bars.
    # White stands out best against the dark blue.
    for i, y in enumerate(ys):
        if len(x) != len(y):
            xi = x[i]
        else:
            xi = x
        l = labels[i] if labels is not None else None
        plt.plot(xi, y, dashes[i], color=colors[i], lw=3, label=l)

    # Limit the range of the plot to only where the data is.
    # Avoid unnecessary whitespace.
    if ylim is not None:
        plt.ylim(*ylim)
    else:
        plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    # Make the title big enough so it spans the entire plot, but don't make it
    # so big that it requires two lines to show.
    plt.title(title, fontsize=16)

    plt.tight_layout()

    if labels is not None:
        # Shrink current axis
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.01, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, framealpha=0.0, fontsize=16)

    # Finally, save the figure as a PNG.
    # You can also save it as a PDF, JPEG, etc.
    # Just change the file extension in this call.
    # bbox_inches="tight" removes all the extra whitespace on the edges of your plot.
    if save_file is not None:
        plt.savefig(save_file,_inches='tight')
    else:
        plt.show()