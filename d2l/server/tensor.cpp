//
// Created by 田地 on 2020/9/26.
//

#include "tensor.h"
#include <string>
#include <stdexcept>

ten::Tensor::Tensor() {}
ten::Tensor::~Tensor(){}
ten::Tensor::Tensor(vector<int> dims, DataType t, vector<char> data) {
    this->dims = dims;
    this->dtype = t;
    this->data = data;
}

const vector<char>& ten::Tensor::Data() const {
    return this->data;
}

const ten::DataType ten::Tensor::Type() const {
    return this->dtype;
}

const vector<int>& ten::Tensor::Dims() const {
    return this->dims;
}