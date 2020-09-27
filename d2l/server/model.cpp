//
// Created by 田地 on 2020/9/18.
//
#include <map>
#include "model.h"
#include "compute.h"

OnnxModel::OnnxModel() {
    this->model = new onnx::ModelProto();
}

OnnxModel::~OnnxModel() {
    cout<< "we now destroy the model" << endl;
}

void OnnxModel::Load(istream *is) {
    this->model->ParseFromIstream(is);
    cout<< "load model successfully" <<endl;
    auto graph = this->model->graph();
    for(int i=0; i<graph.initializer_size(); i++) {
//        cout<< graph.initializer(i).name() <<endl;
        this->inputs[graph.initializer(i).name()] = graph.initializer(i);
    }
    this->eng = engine(engine::kind::cpu, 0);
    this->initPrim();
}

void OnnxModel::initPrim() {
    auto graph = this->model->graph();
    for(int i=0; i<graph.node_size(); i++) {
        auto node = graph.node(i);
        if(node.op_type() == "Conv") {
            this->initConv(node, i==0);
        } else {
            cout<< "invalid type" <<endl;
        }
    }
}

void OnnxModel::initConv(onnx::NodeProto& node, bool isFirst) {
    if(node.op_type() != "Conv") {
        cerr<< "expected Conv node" <<endl;
        exit(1);
    }

    auto inputs = this->inputs;

    vector<int> XDim;
    if(isFirst) {
        auto graph = this->model->graph();
        for(int i=0; i<graph.input_size(); i++) {
            if(node.input(0) == graph.input(i).name()) {
                auto xDims = graph.input(i).type().tensor_type().shape().dim();
                for(int j=0; j<xDims.size(); j++) {
                    XDim.push_back(xDims[j].dim_value());
                }
            }
            break;
        }
    } else {
        auto inputName = node.input(1);
        if(inputs.find(inputName) == inputs.end()) {
            cerr<< "input " << inputName << " not found" <<endl;
            exit(1);
        }
        XDim.assign(inputs.at(inputName).dims().begin(), inputs.at(inputName).dims().end());
    }

    if(inputs.find(node.input(1)) == inputs.end()) {
        cerr<< "input " << node.input(1) << " not found" <<endl;
        exit(1);
    }
    auto W = inputs.at(node.input(1));

    if(inputs.find(node.input(2)) == inputs.end()) {
        cerr<< "input " << node.input(2) << " not found" <<endl;
        exit(1);
    }
    auto B = inputs.at(node.input(2));

//    memory::dims src_tz(XDim.begin(), XDim.end());
//    memory::dims weights_tz(W.dims().begin(), W.dims().end());
//    memory::dims bias_tz (B.dims().begin(), B.dims().end());
    memory::dims src_tz = {1, 1, 28, 28};
    memory::dims weights_tz = {6, 1, 5, 5};
    memory::dims bias_tz = {6};
    memory::dims dst_tz = {1, 6, 28, 28};
    memory::dims strides = {1, 1};
    memory::dims padding = {2, 2};

    auto src_md = memory::desc({src_tz}, dt::f32, tag::any);
    auto bias_md = memory::desc({bias_tz}, dt::f32, tag::any);
    auto weights_md = memory::desc({weights_tz}, dt::f32, tag::any);
    auto dst_md = memory::desc({dst_tz}, dt::f32, tag::any);

    try {
        auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
                                                   algorithm::convolution_direct, src_md, weights_md,
                                                   bias_md, dst_md, strides, padding, padding);
        auto prim_desc = convolution_forward::primitive_desc(conv_desc, this->eng);
        auto conv_prim = convolution_forward(prim_desc);
        this->prims[node.name()] = conv_prim;
        cout<< "init primitive for" << node.name() << "successfully" << endl;
    } catch(dnnl::error e) {
        cout<< e.status <<endl;
        cout<< e.what() <<endl;
    }
}

void * OnnxModel::Forward(const comp::Data& data) {
    auto graph = this->model->graph();
    for(int i=0; i<graph.node_size(); i++) {
        comp::Forward(graph.node(i), data);
    }
    return NULL;
}


