//
// Created by 田地 on 2020/9/26.
//

#include "tensor.h"

ten::Tensor::Tensor() {}
ten::Tensor::~Tensor(){}
ten::Tensor::Tensor(vector<int> dims, DataType t, void* ptr) {
    this->dims = dims;
    this->dtype = t;
    this->ptr = ptr;
}

void* ten::Tensor::Data() const {
    return this->ptr;
}

ten::DataType ten::Tensor::Type() const {
    return this->dtype;
}

vector<int> ten::Tensor::Dims() const {
    return this->dims;
}