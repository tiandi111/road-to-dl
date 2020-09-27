//
// Created by 田地 on 2020/9/18.
//

#ifndef SERVER_MODEL_H
#define SERVER_MODEL_H

#include <istream>
#include <map>
#include "onnx.pb.h"
#include "compute.h"
#include "dnnl.hpp"

using namespace std;
using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

class OnnxModel {
public:
    OnnxModel();
    ~OnnxModel();
    void Load(istream *is);
    void * Forward(const comp::Data& data);
private:
    void initPrim();
    void initConv(onnx::NodeProto& node, bool isFirst);
private:
    onnx::ModelProto* model;
    std::unordered_map<string, onnx::TensorProto> inputs;
    std::unordered_map<string, dnnl::primitive> prims;
    dnnl::engine eng;
};

#endif //SERVER_MODEL_H
