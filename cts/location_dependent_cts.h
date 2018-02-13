//
// Created by Mel Roderick on 8/11/17.
//

#ifndef CONTEXT_WEIGHTING_TREE_LOCATION_DEPENDENT_CWT_SYMBOL_H
#define CONTEXT_WEIGHTING_TREE_LOCATION_DEPENDENT_CWT_SYMBOL_H

#include <tuple>
#include "cts.h"

using namespace std;

class location_dependent_cts {
public:
    location_dependent_cts(int w, int h, int alphabet_size);
    ~location_dependent_cts();
    double process_image(uint8 *image);

    uint8 get_pixel(uint8 *image, int x, int y, int w, int h);
    tuple<double, double> feed_bits_to_tree(cts *tree, uint8 inp_pixel, uint8 *context);
    cts ***build_cwt_array(int w, int h, int alphabet_size);
    void destroy_cwt_array(cts ***cwt_array, int w, int h);
    void process_image_cols(cts ***cwt_array, uint8 *image, tuple<double, double>* result, int from_x, int to_x, int w, int h);

private:
    cts ***m_cwt_array;
    int m_width, m_height;
};

#endif //CONTEXT_WEIGHTING_TREE_LOCATION_DEPENDENT_CWT_SYMBOL_H
