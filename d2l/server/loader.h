//
// Created by 田地 on 2020/9/22.
//

#ifndef SERVER_LOADER_H
#define SERVER_LOADER_H

#include "graph.h"
#include "node.h"
#include "tensor.h"
#include "onnx.pb.h"
#include <istream>
#include <map>

using namespace std;

namespace load {
    grp::Graph LoadOnnx(istream *is);
    node::OpType OnnxType2OpType(string t);
    ten::DataType OnnxDataType2TenDataType(int odtype);
    map<string, vector<int>> ParseInputInfos(onnx::ModelProto oModel);
    map<string, vector<int>> ParseOutputInfos(onnx::ModelProto oModel);
    std::shared_ptr<node::Node> ParseNode(
            onnx::NodeProto& oNode,
            int id,
            const vector<string>& inputs,
            const vector<string>& outputs,
            shared_ptr<grp::Graph> gptr);
    std::shared_ptr<node::ConvNode> ParseConvNode(
            onnx::NodeProto& oNode,
            int id,
            const vector<string>& inputs,
            const vector<string>& outputs,
            shared_ptr<grp::Graph> gptr);
    std::shared_ptr<node::BatchNormNode> ParseBnNode(
            onnx::NodeProto& oNode,
            int id,
            const vector<string>& inputs,
            const vector<string>& outputs,
            shared_ptr<grp::Graph> gptr);
    unordered_map<string, ten::Tensor> ReadWeights(onnx::GraphProto oGraph);
}

#endif //SERVER_LOADER_H
