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
        vector<char> data;
    public:
        Tensor();
        ~Tensor();
        Tensor(vector<int> dims, DataType t, vector<char> data);
        const vector<char>& Data() const;
        const DataType Type() const;
        const vector<int>& Dims() const;
        // todo: finish this element wise multiplcation
//        void Multiply(Tensor& A, vector<int>& Axes, bool InPlace);
    };
}

#endif //SERVER_TENSOR_H
