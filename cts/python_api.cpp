//
// Created by maroderi on 7/23/17.
//

#include "location_dependent_cts.h"

extern "C" location_dependent_cts* construct_cts(int width, int height, int alphabet_size) {
    return new location_dependent_cts(width, height, alphabet_size);
}

extern "C" double psuedo_count_for_image(location_dependent_cts* ldc, uint8 *image) {
    return ldc->process_image(image);
}
