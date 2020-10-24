//
// Created by 田地 on 2020/9/27.
//

#ifndef SERVER_HELPER_H
#define SERVER_HELPER_H

#include <vector>
#include "dnnl.hpp"
#include <iostream>

using namespace std;

dnnl::memory::dims ComputeConvOutputDims(
        int n,
        int h, int w,
        int hk, int wk,
        int hlp, int hhp,
        int wlp, int whp,
        int hs, int ws,
        int oc) {
    dnnl::memory::dims outDims = {
            n, oc,
            (h-hk+hlp+hhp)/hs+1,
            (w-wk+wlp+whp)/ws+1};
    return outDims;
}

vector<int64_t> gatherDims(const vector<int64_t>& srcDims, const vector<int64_t>& indicesDims, int axis) {
    if(axis < 0 || axis >= srcDims.size()) {
        throw std::invalid_argument("axis out of range: " + to_string(axis) + ", [0, " + to_string(srcDims.size()) + "]");
    }
    vector<int64_t> gDims;
    gDims.insert(gDims.end(), srcDims.begin(), srcDims.begin()+axis-1);
    gDims.insert(gDims.end(), indicesDims.begin(), indicesDims.end());
    gDims.insert(gDims.end(), srcDims.begin()+axis+1, srcDims.end());
    return gDims;
}

// |          | axis |          |
//               |
//               V
// |       |  indices  |           |
template<typename T>
void gather(T* src, T* dst, const int64_t* indices, const vector<int64_t>& srcDims, const vector<int64_t>& indicesDims, int axis) {
    if(axis < 0 || axis >= srcDims.size()) {
        throw std::invalid_argument("axis out of range: " + to_string(axis) + ", [0, " + to_string(srcDims.size()) + "]");
    }
    int bfAxisProd = 1;
    int srcCpySize = 1;
    int axisSize = srcDims[axis];
    for(int i=0; i<srcDims.size(); i++) {
        if(i<axis) bfAxisProd *= srcDims[i];
        if(i>axis) srcCpySize *= srcDims[i];
    }
//    srcCpySize *= sizeof(T);
    int idxNumOfEle = 1;
    int idxUnitSize = indicesDims[indicesDims.size()-1];
    for(int i=0; i<indicesDims.size()-1; i++) {
        idxNumOfEle *= indicesDims[i];
    }
    for(int i=0; i<bfAxisProd; i++) {
        for(int j=0; j<idxNumOfEle; j++) {
            for(int k=0; k<idxUnitSize; k++) {
                int64_t idx = indices[j*idxUnitSize + k];
                if(idx < 0 || idx >= srcDims[axis]) {
                    throw std::invalid_argument("index out of range: " + to_string(idx) + ", [0, " + to_string(srcDims[axis]) + "]");
                }
                // index in src: src + i * axisSize * srcCpySize
                // offset in src: idx * srcCpySize
                void* st = src + i * axisSize * srcCpySize + idx * srcCpySize;
                memcpy(dst, st, srcCpySize*sizeof(T));
                dst += srcCpySize;
            }
        }
    }
}

#endif //SERVER_HELPER_H
