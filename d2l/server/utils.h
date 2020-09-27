//
// Created by 田地 on 2020/9/27.
//

#ifndef SERVER_HELPER_H
#define SERVER_HELPER_H

#include <vector>
#include "dnnl.hpp"

using namespace std;

//vector<int>& ComputeConvOutputDims(
//        int n,
//        int h, int w,
//        int hk, int wk,
//        int hlp, int hhp,
//        int wlp, int whp,
//        int hs, int ws,
//        int oc) {
//    vector<int> outDims = {
//            n, oc,
//            (h-hk+hlp+hhp)/hs+1,
//            (w-wk+wlp+whp)/ws+1};
//    return outDims;
//}

dnnl::memory::dims& ComputeConvOutputDims(
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

#endif //SERVER_HELPER_H
