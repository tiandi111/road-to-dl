//
// Created by 田地 on 2020/9/24.
//

#include "loader.h"
#include <fstream>

// parse input dimension info
map<string, vector<int>> load::ParseInputInfos(onnx::ModelProto oModel) {
    map<string, vector<int>> inputDimInfos;
    for (int i = 0; i < oModel.graph().input_size(); i++) {
        cout << "the " << i << " th input, dimension: ";
        vector<int> inDims;
        auto oInDims = oModel.graph().input(i).type().tensor_type().shape().dim();
        for (int j = 0; j < oInDims.size(); j++) {
            inDims.push_back(oInDims[j].dim_value());
            cout << oInDims[j].dim_value() << " ";
        }
        cout << endl;
        inputDimInfos[oModel.graph().input(i).name()] = inDims;
    }
    return inputDimInfos;
}

// parse output dimension info
map<string, vector<int>> load::ParseOutputInfos(onnx::ModelProto oModel) {
    map<string, vector<int>> outputDimInfos;
    for(int i=0; i<oModel.graph().output_size(); i++) {
        cout<< "the " << i << " th output, dimension: ";
        vector<int> outDims;
        auto oOutDims = oModel.graph().output(i).type().tensor_type().shape().dim();
        for(int j=0; j<oOutDims.size(); j++) {
            outDims.push_back(oOutDims[j].dim_value());
            cout<< oOutDims[j].dim_value() << " ";
        }
        cout<<endl;
        outputDimInfos[oModel.graph().output(i).name()] = outDims;
    }
    return outputDimInfos;
}

node::OpType load::OnnxType2OpType(string t) {
    if(t == "Conv") {
        return node::conv;
    }
    if(t == "BatchNormalization") {
       return node::bn;
    }
    return node::conv;
}

ten::DataType load::OnnxDataType2TenDataType(int odtype) {
    switch(odtype) {
        case onnx::TensorProto::FLOAT :
            return ten::DataType::f32;
        case onnx::TensorProto::INT64 :
            return ten::DataType::i64;
        default:
            throw "unsupported data type";
            return ten::DataType::unknown;
    }
}

node::Node* load::ParseNode(
        onnx::NodeProto oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph* g) {
    if(oNode.op_type() == "Conv") {
        return ParseConvNode(oNode, id, inputs, outputs, g);
    }
    node::Node* node = new node::Node(load::OnnxType2OpType(oNode.op_type()), id, inputs, outputs, g);
    return node;
}

node::ConvNode* load::ParseConvNode(
        onnx::NodeProto oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph* g) {
    int group;
    vector<int> dilations;
    vector<int> kernelShape;
    vector<int> pads; // l, r, u, d
    vector<int> strides;
    for(int i=0; i<oNode.attribute_size(); i++) {
        auto attr = oNode.attribute(i);
        if(attr.name() == "dilations") {
            dilations.assign(attr.ints().begin(), attr.ints().end());
        }
        if(attr.name() == "kernel_shape") {
            kernelShape.assign(attr.ints().begin(), attr.ints().end());
        }
        if(attr.name() == "pads") {
            pads.assign(attr.ints().begin(), attr.ints().end());
        }
        if(attr.name() == "strides") {
            strides.assign(attr.ints().begin(), attr.ints().end());
        }
        if(attr.name() == "group") {
            group = attr.i();
        }
    }
    return new node::ConvNode(
            load::OnnxType2OpType(oNode.op_type()),
            id,
            inputs,
            outputs,
            g,
            group,
            dilations,
            kernelShape,
            pads,
            strides
            );
}

unordered_map<string, ten::Tensor> load::ReadWeights(onnx::GraphProto oGraph) {
    unordered_map<string, ten::Tensor> weights;
    for(int i=0; i<oGraph.initializer_size(); i++) {
        auto initial = oGraph.initializer(i);
        vector<int> dims(initial.dims().begin(), initial.dims().end());
        weights[initial.name()] = ten::Tensor(
                dims,
                load::OnnxDataType2TenDataType(initial.data_type()),
                initial.mutable_raw_data());

        cout<<  initial.name() << "," << initial.data_type() <<endl;
        for(int i=0; i<dims.size(); i++) {
            cout<< dims[i];
        }
        cout<<endl;
    }
    return weights;
}


grp::Graph* load::LoadOnnx(istream *is) {

    onnx::ModelProto oModel = onnx::ModelProto();
    oModel.ParseFromIstream(is);
    cout<< "load onnx model successfully" <<endl;

    map<string, node::Node*> parentNodeMap;

    grp::Graph* graph = new grp::Graph(ParseInputInfos(oModel), ParseOutputInfos(oModel), load::ReadWeights(oModel.graph()));

    // parse node info
    for(int i=0; i<oModel.graph().node_size(); i++) {
        auto oNode = oModel.graph().node(i);

//        node::Node * node = new node::Node(load::OnnxType2OpType(oNode.op_type()), i, graph);
        const vector<string> inputs(oNode.input().begin(), oNode.input().end());
        const vector<string> outputs(oNode.output().begin(), oNode.output().end());
        node::Node* node = load::ParseNode(oNode, i, inputs, outputs, graph);
        if(i == 0) {
            graph->SetRoot(node);
        }

        graph->AddNode(*node);

        for(int j=0; j<oNode.output_size(); j++) {
            parentNodeMap[oNode.output(j)] = node;
        }

        for(int j=0; j<oNode.input_size(); j++) {
            string inputName = oNode.input(j);
            if(parentNodeMap.find(inputName) != parentNodeMap.end()) {
                parentNodeMap[inputName]->AddSucc(node);
            }
        }

    }

    return graph;
}

int main() {
//    ifstream in("test_conv.onnx", ios_base::binary);
    ifstream in("/Users/tiandi03/road-to-dl/d2l/lenet.onnx", ios_base::binary);
    auto g = load::LoadOnnx(&in);
    auto nodes = g->GetNodes();
//    for(int i=0; i<nodes.size(); i++) {
//        cout<< nodes[i].Type() <<endl;
//    }
}