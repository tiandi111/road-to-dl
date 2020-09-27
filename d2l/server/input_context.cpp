//
// Created by 田地 on 2020/9/25.
//

#include "input_context.h"

ictx::InputContext::InputContext() {}

ictx::InputContext::~InputContext() {}

ictx::InputContext::InputContext(unordered_map<string, ten::Tensor>& inputs) {
    this->inputs = inputs;
}

const unordered_map<string, ten::Tensor>& ictx::InputContext::Inputs() {
    return this->inputs;
}
