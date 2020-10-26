//
// Created by 田地 on 2020/9/26.
//

#include "tensor.h"
#include <string>
#include <stdexcept>
#include <iostream>

void ten::Tensor::computeBytes() {
    int64_t b = 1;
    for(int64_t d : dims) {b *= d;}
    switch (dtype) {
        case f32:
            bytes = sizeof(float) * b;
            return;
        case i64:
            bytes = sizeof(int64_t) * b;
            return;
        case i8:
            bytes = sizeof(int8_t) * b;
            return;
        default:
            throw std::invalid_argument("unsupported data type");
    }
}

void ten::Tensor::checkBuffer() {
    if(data.size() != bytes) {
        throw std::invalid_argument("buffer size does not match dimsions");
    }
}


ten::Tensor::Tensor(vector<int64_t> dims, DataType t) : dims(dims), dtype(t) {
    computeBytes();
    data = vector<char>(bytes);
}

ten::Tensor::Tensor(vector<int64_t> dims, DataType t, vector<char>& data): dims(dims), dtype(t), data(data) {
    computeBytes();
    checkBuffer();
}

void ten::Tensor::Write(void* handle) {
    char* p = (char*) handle;
    data.assign(p, p+bytes);
}


const vector<char>& ten::Tensor::Data() const {
    return data;
}

const ten::DataType ten::Tensor::Type() const {
    return dtype;
}

const vector<int64_t>& ten::Tensor::Dims() const {
    return dims;
}