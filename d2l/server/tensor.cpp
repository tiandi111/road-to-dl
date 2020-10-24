//
// Created by 田地 on 2020/9/26.
//

#include "tensor.h"
#include <string>
#include <stdexcept>

const vector<char>& ten::Tensor::Data() const {
    return this->data;
}

const ten::DataType ten::Tensor::Type() const {
    return this->dtype;
}

const vector<int64_t>& ten::Tensor::Dims() const {
    return this->dims;
}