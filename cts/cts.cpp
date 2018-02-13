//
// Created by Mel Roderick on 8/11/17.
//

#include <iostream>
#include <cmath>
#include "cts.h"

void kill_tree_recursive(cts_node *ptr) {
    for (int i = 0; i < ptr->m_alphabet_size; ++i) {
        if (ptr->m_children[i] != 0) {
            kill_tree_recursive(ptr->m_children[i]);
        }
    }
    delete ptr;

}

cts::cts(int context_depth, int alphabet_size) {
    m_context_depth = context_depth;
    m_alphabet_size = alphabet_size;
    m_root_node = new cts_node(this, 0, context_depth, alphabet_size);
    m_n = 0;
    m_log_1_minus_alpha, m_log_alpha = 0;
}

cts::~cts() {
    kill_tree_recursive(m_root_node);
};

double cts::logprob(uint8 *context, uint8 symbol) {
    check_input(context, symbol);

    return m_root_node->update(context, symbol, false);
}

double cts::update_and_logprob(uint8 *context, uint8 symbol) {
    check_input(context, symbol);

    m_n += 1.0;
    m_log_alpha = std::log(1.0 / (m_n + 1.0));
    m_log_1_minus_alpha = std::log(m_n / (m_n + 1.0));

    return m_root_node->update(context, symbol, true);
}

void cts::check_input(uint8 *context, uint8 symbol) {
    if (symbol >= m_alphabet_size) goto error;
    for (int i = 0; i < m_context_depth; ++i) {
        if (context[i] >= m_alphabet_size) {
            goto error;
        }
    }
    return;

    error:
    std::cerr << "Assertion Failed: symbol must be less than alphabet_size. \n"
                         "In " << __FILE__ << " Line: " << __LINE__ << "\n";
    std::terminate();
}