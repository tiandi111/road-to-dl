//
// Created by 田地 on 2020/9/25.
//

#include "engine.h"
#include "mkl.h"
#include "utils.h"

eng::Engine::Engine() {}

eng::Engine::Engine(string name, DeviceType t) {
    this->name = name;
    this->dtype = t;
}

eng::Engine::~Engine() {}

//eng::Engine::Execute(ictx::InputContext ctx, grp::Graph g) {
//
//}

eng::MKLEngine::MKLEngine(string name, DeviceType t) : Engine(name, t){
    // todo: engine index?
    this->eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
}

void eng::MKLEngine::Execute(ictx::InputContext ctx, grp::Graph* g) {
    if(!g) {
        throw "nil graph";
    }
    dnnl::stream stream(this->eng);
    // inputs and weights, for quick getting only
    unordered_map<string, dnnl::memory> inputs;
    // create memory object for weights
    for(auto& it : g->GetWeights()) {
        auto tDims = it.second.Dims();
        dnnl::memory::dims dims(tDims.begin(), tDims.end());
        auto wMemory  = dnnl::memory({dims, dt::f32, tag::nchw}, this->eng); // todo: data type and tag
//        dnnl::write_to_dnnl_memory(it.second.Data(), wMemory);
        inputs.insert({it.first, wMemory});
    }
    // create memory object for inputs
    for(auto& it : ctx.Inputs()) {
        dnnl::memory::dims dims(it.second.Dims().begin(), it.second.Dims().end());
        auto inMemory = dnnl::memory({dims, dt::f32, tag::nchw}, this->eng); // todo: data type and tag
//        dnnl::write_to_dnnl_memory(it.second.Data(), inMemory);
        inputs.insert({it.first, inMemory});
    }
    auto execCtx = MKLExecutionContext(
            stream,
            this->eng,
            inputs,
            g);
}

eng::MKLExecutionContext::MKLExecutionContext(
        dnnl::stream stream,
        dnnl::engine eng,
        unordered_map<string, dnnl::memory>& inputs,
        grp::Graph* g) : inputs(inputs) {
    if(!g) {
        throw "nil pointer g";
    }
    this->stream = stream;
    this->eng = eng;
    this->g = g;
}

void eng::MKLExecutionContext::Execute() {
    auto nodes = this->g->GetNodes();
    auto& prims = this->prims;
    auto& args = this->args;
    for(int i=0; i<nodes.size(); i++) {
        InitNode(nodes[i]);
    }
    for(int i=0; i<this->prims.size(); i++) {
        prims[i].execute(this->stream, args[i]);
    }
}

void eng::MKLExecutionContext::InitNode(const node::Node& n) {
    switch(n.Type()) {
        case node::OpType::conv :
            auto convn = n;
            InitConvNode(n);
            break;
        default:
            throw "invalid op type";
    }
}

void eng::MKLExecutionContext::InitConvNode(const node::ConvNode& node) {
    auto& inputs = this->inputs;
    auto g = this->g;
    auto inputList = node.Inputs();
    if(inputList.size() > 3 || inputList.size() < 2) {
        throw "too many inputs, required 3";
    }
    vector<int> srcDims; dnnl::memory srcMemory;
    vector<int> wDims; dnnl::memory wMemory;
    vector<int> bDims; dnnl::memory bMemory;
    for(int i=0; i<inputList.size(); i++) {
        auto name = inputList[i];
        auto got = g->GetWeights().find(name);
        auto mem = inputs.find(name);
        if(got == g->GetWeights().end() && mem == inputs.end()) {
            throw "input not found, the model may have been broken";
        }
        if(name.rfind("weight") != string::npos) {
            wDims = got.Dims();
            wMemory = mem;
        } else if(name.rfind("bias") != string::npos) {
            bDims = got.Dims();
            bMemory = mem;
        } else {
            auto dnnlSrcDims = mem.dims();
            srcDims.assign(dnnlSrcDims.begin(), dnnlSrcDims.end());
            srcMemory = mem;
        }
    }
    auto dstDims = ComputeConvOutputDims(
            srcDims[0], srcDims[2], srcDims[3], node.KernelShape()[0], node.KernelShape()[1],
            node.Pads()[0], node.Pads()[1], node.Pads()[2], node.Pads()[3], node.Strides()[0], node.Strides()[1],
            wDims[0]);
    auto dstMemory = dnnl::memory({dstDims, dt::f32, tag::nchw}, this->eng); // todo: data type
    inputs.insert({node.Outputs()[0], dstMemory});
    // todo: if bias not exist?
    vector<int> vDstDims(dstDims.begin(), dstDims.end());
    auto prim = mkl::CnnPrimitive(
            this->eng,
            this->stream,
            srcDims,
            wDims,
            bDims,
            vDstDims,
            node.Strides(),
            node.Pads());
    this->prims.push_back(prim);
    this->args.push_back({
         {DNNL_ARG_SRC, srcMemory},
         {DNNL_ARG_WEIGHTS, wMemory},
         {DNNL_ARG_BIAS, bMemory},
         {DNNL_ARG_DST, dstMemory}
    });
}

int main() {

}
