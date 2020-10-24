//
// Created by 田地 on 2020/9/26.
//

#ifndef SERVER_TENSOR_H
#define SERVER_TENSOR_H

#include <vector>

using namespace std;

namespace ten {
    enum DataType {
        unknown,
        f32,
        i64,
        i8,
    };

    class Tensor {
    private:
        vector<int64_t> dims;
        DataType dtype;
        vector<char> data;
    public:
        Tensor() = default;
        ~Tensor() = default;
        Tensor(vector<int64_t>& dims, DataType t, vector<char>& data) :
                dims(dims), dtype(t), data(data) {};
        const vector<char>& Data() const;
        const DataType Type() const;
        const vector<int64_t>& Dims() const;
        // todo: finish this element wise multiplcation
//        void Multiply(Tensor& A, vector<int>& Axes, bool InPlace);
    };
}

#endif //SERVER_TENSOR_H

