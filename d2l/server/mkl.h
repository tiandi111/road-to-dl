//
// Created by 田地 on 2020/9/25.
//

#ifndef SERVER_MKL_H
#define SERVER_MKL_H

#include "dnnl.hpp"
#include <vector>

using namespace std;
using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace mkl {

    primitive CnnPrimitive(
            const engine& eng,
            const stream& stream,
            const vector<int>& srcDims,
            const vector<int>& wDims,
            const vector<int>& bDims,
            const vector<int>& dstDims,
            const vector<int>& strides,
            const vector<int>& padding);

}

#endif //SERVER_MKL_H
