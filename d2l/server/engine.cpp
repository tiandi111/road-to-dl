//
// Created by 田地 on 2020/9/25.
//

#include "engine.h"
#include "mkl.h"
#include "utils.h"
#include <memory>
#include <stdexcept>

eng::Engine::Engine() {}

eng::Engine::Engine(string name, DeviceType t) {
    this->name = name;
    this->dtype = t;
}

eng::MKLEngine::MKLEngine(string name, DeviceType t) : Engine(name, t){
    // todo: engine index?
    this->eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
}

void eng::MKLEngine::Execute(ictx::InputContext& ctx, grp::Graph& g) {
    dnnl::stream stream(this->eng);
    // inputs and weights, for quick getting only
    unordered_map<string, dnnl::memory> inputs;
    // create memory object for inputs
    for(auto& it : ctx.Inputs()) {
        dnnl::memory::dims dims(it.second.Dims().begin(), it.second.Dims().end());
        auto inMemory = dnnl::memory({dims, dt::f32, tag::nchw}, this->eng); // todo: data type and tag
        mkl::WriteToDnnlMemory(it.second.Data().data(), inMemory);
        inputs.insert({it.first, inMemory});
    }
    auto execCtx = MKLExecutionContext(
            stream,
            this->eng,
            inputs,
            g);
    execCtx.Execute();
}

eng::MKLExecutionContext::MKLExecutionContext(
        dnnl::stream& stream,
        dnnl::engine& eng,
        unordered_map<string, dnnl::memory>& inputs,
        grp::Graph& g) : inputs(inputs), stream(stream), eng(eng), g(g) {
}

void eng::MKLExecutionContext::Execute() {
    auto& nodes = this->g.GetNodes();
    auto& prims = this->prims;
    auto& args = this->args;
    for(const auto & node : nodes) {
        InitNode(node);
    }
    for(int i=0; i<this->prims.size(); i++) {
        prims[i].execute(this->stream, args[i]);
    }
    this->stream.wait();
    float out[32];
    mkl::ReadFromDnnlMemory(out, this->args[0][17]);
    for(float o : out) {
        cout<< o <<endl;
    }
}

void eng::MKLExecutionContext::InitNode(std::shared_ptr<node::Node> n) {
    switch(n->Type()) {
        case node::OpType::conv : {
            std::shared_ptr<node::ConvNode> node = std::dynamic_pointer_cast<node::ConvNode>(n);
            InitConvNode(node);
            break;
        }
        case node::OpType::bn : {
            std::shared_ptr<node::BatchNormNode> node = std::dynamic_pointer_cast<node::BatchNormNode>(n);
            InitBnNode(node);
            break;
        }
        default:
            throw std::runtime_error("invalid op type" + to_string(n->Type()));
    }
}

void eng::MKLExecutionContext::InitConvNode(std::shared_ptr<node::ConvNode> node) {
    auto& inputs = this->inputs;
    auto graph = node->GetGraph();
    // get weight info
    dnnl::memory::dims wDims(node->WeightDims().begin(), node->WeightDims().end());
    // notice that the tag should be compatible with actual dims
    auto wMemory  = dnnl::memory({wDims, dt::f32, tag::nchw}, this->eng);
    mkl::WriteToDnnlMemory(graph->GetWeightHandle(node->WeightName()), wMemory);
    // get bias info
    dnnl::memory::dims bDims(node->BiasDims().begin(), node->BiasDims().end());
    auto bMemory  = dnnl::memory({bDims, dt::f32, tag::x}, this->eng);
    mkl::WriteToDnnlMemory(graph->GetWeightHandle(node->BiasName()), bMemory);
    // get src info
    auto srcMemoryPair = inputs.find(node->SrcInputName());
    if(srcMemoryPair == inputs.end()) {
        throw std::runtime_error("source " + node->SrcInputName() + " not found");
    }
    auto srcMemory = srcMemoryPair->second;
    auto oSrcDims = srcMemory.get_desc().dims();
    vector<int> srcDims(oSrcDims.begin(), oSrcDims.end());

    auto dstDims = ComputeConvOutputDims(
            srcDims[0], srcDims[2], srcDims[3], node->KernelShape()[0], node->KernelShape()[1],
            node->Pads()[0], node->Pads()[1], node->Pads()[2], node->Pads()[3], node->Strides()[0], node->Strides()[1],
            node->WeightDims()[0]);
    auto dstMemory = dnnl::memory({dstDims, dt::f32, tag::nchw}, this->eng); // todo: data type
    inputs.insert({node->Outputs()[0], dstMemory});
    // todo: if bias not exist?
    vector<int> vDstDims(dstDims.begin(), dstDims.end());

    memory::dims src_dims(srcDims.begin(), srcDims.end());
    memory::dims weights_dims(wDims.begin(), wDims.end());
    memory::dims bias_dims(bDims.begin(), bDims.end());
    memory::dims dst_dims(dstDims.begin(), dstDims.end());
    // Strides, padding dimensions.
    memory::dims strides_dims(node->Strides().begin(), node->Strides().end());
    memory::dims padding_dims_l;
    memory::dims padding_dims_r;
    for(int i=0; i<node->Pads().size(); i+=2) {
        padding_dims_l.push_back(node->Pads()[i]);
        padding_dims_r.push_back(node->Pads()[i+1]);
    }
    // Create memory descriptors with format_tag::any for the primitive. This
    // enables the convolution primitive to choose memory layouts for an
    // optimized primitive implementation, and these layouts may differ from the
    // ones provided by the user.
    auto conv_src_md = memory::desc(src_dims, dt::f32, tag::nchw);
    auto conv_weights_md = memory::desc(weights_dims, dt::f32, tag::nchw);
    auto conv_dst_md = memory::desc(dst_dims, dt::f32, tag::nchw);
    // Create memory descriptor and memory object for input bias.
    auto user_bias_md = memory::desc(bias_dims, dt::f32, tag::a);
    // Create operation descriptor.
    auto conv_desc = convolution_forward::desc(prop_kind::forward_training,
                                               algorithm::convolution_direct, conv_src_md, conv_weights_md,
                                               user_bias_md, conv_dst_md, strides_dims, padding_dims_l,
                                               padding_dims_r);
    primitive_attr conv_attr;
    // Create primitive descriptor.
    auto conv_pd = convolution_forward::primitive_desc(conv_desc, conv_attr, eng);
    // Create the primitive.
    auto prim = convolution_forward(conv_pd);
    // In case dnnl creates a different format from the user defined one, we need to reorder them
    auto convSrcMem = srcMemory;
    auto convWeightsMem = wMemory;
    auto convDstMem = dstMemory;
    // reorder source memory
    if (conv_pd.src_desc() != srcMemory.get_desc()) {
        convSrcMem = memory(conv_pd.src_desc(), this->eng);
        reorder(srcMemory, convSrcMem)
                .execute(this->stream, srcMemory, convSrcMem);

    }
    // reorder weight memory
    if (conv_pd.weights_desc() != wMemory.get_desc()) {
        convWeightsMem = memory(conv_pd.weights_desc(), this->eng);
        reorder(wMemory, convWeightsMem)
                .execute(this->stream, wMemory, convWeightsMem);
    }
    // reorder destination memory
    if (conv_pd.dst_desc() != dstMemory.get_desc()) {
        convDstMem = memory(conv_pd.dst_desc(), this->eng);
    }

    this->prims.push_back(prim);
    this->args.push_back({
             {DNNL_ARG_SRC, convSrcMem},
             {DNNL_ARG_WEIGHTS, convWeightsMem},
             {DNNL_ARG_BIAS, bMemory},
             {DNNL_ARG_DST, convDstMem}
     });
}

void eng::MKLExecutionContext::InitBnNode(std::shared_ptr<node::BatchNormNode> node) {
    auto graph = node->GetGraph();
    // retrieve srouce memory
    auto srcMemPair = inputs.find(node->SrcInputName());
    if(srcMemPair == inputs.end()) {
        throw std::runtime_error("source " + node->SrcInputName() + " not found");
    }
    auto srcMem = srcMemPair->second;
    // scale and shift memory
    auto scaleShiftMd = memory::desc({2, node->Dim()[0]}, dt::f32, tag::nc);
    auto scaleShiftMem = memory(scaleShiftMd, this->eng);
    size_t scaleShiftSize = scaleShiftMd.get_size();
    mkl::WriteToDnnlMemoryFromTo(graph->GetWeightHandle(node->WightName()), scaleShiftMem,
            0, 0, scaleShiftSize/2);
    mkl::WriteToDnnlMemoryFromTo(graph->GetWeightHandle(node->BiasName()), scaleShiftMem,
            0, scaleShiftSize/2, scaleShiftSize/2);
    // Create operation descriptor.
    auto bnormDesc = batch_normalization_forward::desc(
            prop_kind::forward_inference, srcMem.get_desc(), node->Epsilon(),
                normalization_flags::use_scale_shift
                | normalization_flags::use_global_stats);
    // Create primitive descriptor.
    auto bnormPD
            = batch_normalization_forward::primitive_desc(bnormDesc, this->eng);
    // Create memory objects using memory descriptors created by the primitive
    // descriptor: mean, variance, workspace.
    // NOTE: Here, the ReLU post-ops require a workspace for later usage in
    // backward propagation mode.
    auto meanMem = memory(bnormPD.mean_desc(), this->eng);
    mkl::WriteToDnnlMemory(graph->GetWeightHandle(node->MeanName()), meanMem);
    auto varianceMem = memory(bnormPD.variance_desc(), this->eng);
    mkl::WriteToDnnlMemory(graph->GetWeightHandle(node->VarName()), varianceMem);
    // Create the primitive.
    auto prim = batch_normalization_forward(bnormPD);
    this->prims.push_back(prim);
    this->args.push_back({
            {DNNL_ARG_SRC, srcMem},
            {DNNL_ARG_MEAN, meanMem},
            {DNNL_ARG_VARIANCE, varianceMem},
            {DNNL_ARG_SCALE_SHIFT, scaleShiftMem},
            {DNNL_ARG_DST, srcMem}
     });
}
