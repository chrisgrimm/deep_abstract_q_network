//
// Created by Mel Roderick on 8/11/17.
//

#ifndef CONTEXT_WEIGHTING_TREE_CWT_NODE_SYMBOL_H
#define CONTEXT_WEIGHTING_TREE_CWT_NODE_SYMBOL_H


#include <tiff.h>
class cts;

class cts_node {
public:
    cts_node(cts* model, int depth, int max_depth, int alphabet_size);
    ~cts_node();
    double update(uint8 *context, uint8 symbol, bool update);

    double mix_prediction(double log_p, double log_child_p);
    void update_switching_weights(double log_p, double log_child_p);
    bool is_leaf();

    double* m_counts;
    double m_count_total;
    cts_node** m_children;
    double m_log_stay_prob, m_log_split_prob;
    cts* m_model;
    int m_depth;
    int m_max_depth;
    int m_alphabet_size;
};

inline double log_add(double log_x, double log_y);

#endif //CONTEXT_WEIGHTING_TREE_CWT_NODE_SYMBOL_H
