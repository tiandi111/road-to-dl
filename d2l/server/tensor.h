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
        vector<int> dims;
        DataType dtype;
        void* ptr;
    public:
        Tensor();
        ~Tensor();
        Tensor(vector<int> dims, DataType t, void* ptr);
        void* Data() const;
        DataType Type() const;
        vector<int>& Dims() const;
    };
}

#endif //SERVER_TENSOR_H
