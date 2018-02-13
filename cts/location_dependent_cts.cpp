//
// Created by Mel Roderick on 8/11/17.
//

#include <cmath>
#include "location_dependent_cts.h"


#define UseThreads false
#define NumThreads 2


location_dependent_cts::location_dependent_cts(int w, int h, int alphabet_size) {
    m_cwt_array = build_cwt_array(w, h, alphabet_size);
    m_width = w;
    m_height = h;
}
location_dependent_cts::~location_dependent_cts() {
    destroy_cwt_array(m_cwt_array, m_width, m_height);
}

double location_dependent_cts::process_image(uint8 *image) {
    int w = m_width, h = m_height;

    double cum_prob = 0.0;
    double cum_recoding_prob = 0.0;

#if UseThreads
    std::tuple<double, double> ret[NumThreads];
    std::thread threads[NumThreads];
    int cols_per_thread = w/NumThreads;

    for (int i=0; i < NumThreads; ++i) {
        int from_x = i * cols_per_thread;
        int to_x = std::max(from_x + cols_per_thread, w);
        threads[i] = std::thread(&process_image_cols,
                                 m_cwt_array, image, ret + i, from_x, to_x, w, h, num_bits);
    }
    for (int i=0; i < NumThreads; ++i) {
        threads[i].join();
        tuple<double, double> result = ret[i];
        cum_prob += std::get<0>(result);
        cum_recoding_prob += std::get<1>(result);
    }
#else
    tuple<double, double> result;
    process_image_cols(m_cwt_array, image, &result, 0, w, w, m_height);
    cum_prob += std::get<0>(result);
    cum_recoding_prob += std::get<1>(result);
#endif

    cum_prob = exp(cum_prob);
    cum_recoding_prob = exp(cum_recoding_prob);
    if (cum_prob >= cum_recoding_prob) {
        return 0;
    } else {
        double pseudocount = cum_prob*(1 - cum_recoding_prob)/(cum_recoding_prob - cum_prob);
        return pseudocount;
    }
}

uint8 location_dependent_cts::get_pixel(uint8 *image, int x, int y, int w, int h) {
    if (x < w && x >= 0 && y < h && y >= 0) {
        return image[w*y + x];
    } else {
        return 0;
    }
}

tuple<double, double> location_dependent_cts::feed_bits_to_tree(cts *tree, uint8 inp_pixel, uint8 *context) {
    double bit_prob = tree->update_and_logprob(context, inp_pixel);
    double recoding_prob = tree->logprob(context, inp_pixel);
    return make_tuple(bit_prob, recoding_prob);
}


cts ***location_dependent_cts::build_cwt_array(int w, int h, int alphabet_size) {
    cts ***cwt_array = new cts**[w];
    int context_depth = 4;
    for (int x=0; x<w; x++) {
        cwt_array[x] = new cts*[h];
        for (int y=0; y<h; y++) {
            cwt_array[x][y] = new cts(context_depth, alphabet_size);
        }
    }
    return cwt_array;
}


void location_dependent_cts::destroy_cwt_array(cts ***cwt_array, int w, int h) {
    for (int x=0; x<w; x++) {
        for (int y=0; y<h; y++) {
            delete cwt_array[x][y];
        }
        delete cwt_array[x];
    }
    delete cwt_array;
}


void location_dependent_cts::process_image_cols(cts ***cwt_array, uint8 *image, tuple<double, double>* result, int from_x, int to_x, int w, int h) {
    double cum_prob = 0.0;
    double cum_recoding_prob = 0.0;

    for (int x=from_x; x < to_x; x++) {
        for (int y = 0; y < h; y++) {
            cts *tree = cwt_array[x][y];
            uint8 pixel_value = get_pixel(image, x, y, w, h);
            uint8 context[4];
            context[0] = get_pixel(image, x, y - 1, w, h);
            context[1] = get_pixel(image, x - 1, y, w, h);
            context[2] = get_pixel(image, x - 1, y - 1, w, h);
            context[3] = get_pixel(image, x + 1, y - 1, w, h);
            auto prob_pair = feed_bits_to_tree(tree, pixel_value, context);
            double bit_prob = std::get<0>(prob_pair);
            double recoding_prob = std::get<1>(prob_pair);
            cum_prob += bit_prob;
            cum_recoding_prob += recoding_prob;
        }
    }
    *result = make_tuple(cum_prob, cum_recoding_prob);
}