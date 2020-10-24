//
// Created by 田地 on 2020/9/25.
//

#ifndef SERVER_ENGINE_H
#define SERVER_ENGINE_H

#include "dnnl.hpp"
#include "graph.h"
#include "mkl.h"
#include "input_context.h"
#include "tensor.h"
#include "utils.h"
#include <string>
#include <vector>
#include <iostream>

using namespace std;
using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

namespace eng {
    enum DeviceType {
        cpu,
        gpu
    };

    class Engine {
    private:
        string name;
        DeviceType dtype;

    public:
        Engine();
        Engine(string name, DeviceType t);
        virtual ~Engine() = default;
//        Execute(ictx::InputContext ctx, grp::Graph g);
    };

    class MKLEngine : Engine {
    private:
        std::unordered_map<string, dnnl::primitive> prims;
        dnnl::engine eng;
    public:
        MKLEngine(string name, DeviceType t);
        virtual ~MKLEngine() = default;
        void Execute(ictx::InputContext& ctx, grp::Graph& g);
    };

    // todo: handle shape, gather and others that may not need or not support by mkl prims
    //     define a executable node to proxy mkl primitive and native node operation
    //     before shape, we need original shape, not the shape converted by mkl
    //     before gather, we need to reorder mkl prim shape to the original one
    //     => we need the control of layout transformation, for joint optimization of graph-level and op-level
    //          make it a node!
    //          at this stage, don't do any layout trans, the first goal is to run the lenet!
    class ExecutableNode {
    public:
        virtual ~ExecutableNode() = default;
        virtual void Execute() = 0;
    };

    // todo: maybe we should let all these nodes to init by themselves, think about it
    // nodes that executed natively
    class ShapeNode : public ExecutableNode {
    private:
        dnnl::memory& data;
        dnnl::memory& dst;
    public:
        ShapeNode(dnnl::memory& data, dnnl::memory& dst) : data(data), dst(dst) {};
        inline void Execute() override {
            mkl::WriteToDnnlMemory(data.get_desc().dims().data(), dst);
        }
    };

    class GatherNode : public ExecutableNode {
    private:
        int axis;
        dnnl::memory& data;
        const ten::Tensor& indices;
        dnnl::memory dst;
    public:
        GatherNode(int axis,
                dnnl::memory& data,
                const ten::Tensor& indices,
                dnnl::engine& eng);
        inline dnnl::memory& GetDstMemory() {return this->dst;}
        void Execute() override ;
    };

    // a class of node that executes with the help of mkl
    class ExecNodeMKL : public ExecutableNode {
    protected:
        dnnl::stream& stream;
        dnnl::primitive& prim;
        unordered_map<int, dnnl::memory>& args;
    public:
        ExecNodeMKL(dnnl::stream& stream,
                    dnnl::primitive& prim,
                    unordered_map<int,dnnl::memory>& args) :
                    stream(stream), prim(prim), args(args) {};
        ~ExecNodeMKL() override = default;
        inline void Execute() override {
            this->prim.execute(this->stream, this->args);
        }
    };
    class ExecConvNode : public ExecNodeMKL {
    public:
        ExecConvNode(dnnl::stream& stream,
                dnnl::primitive& prim,
                unordered_map<int,dnnl::memory>& args) :
                ExecNodeMKL(stream, prim, args) {};
    };
    class ExecBNNode : public ExecNodeMKL {
    public:
        ExecBNNode(dnnl::stream& stream,
                dnnl::primitive& prim,
                unordered_map<int,dnnl::memory>& args) :
                ExecNodeMKL(stream, prim, args) {};
    };

    class MKLExecutionContext {
    private:
        dnnl::stream& stream;
        dnnl::engine& eng;
        unordered_map<string, dnnl::memory>& inputs;
        vector<std::shared_ptr<eng::ExecutableNode>> execNodes;
        grp::Graph& g;
    public:
        MKLExecutionContext(
                dnnl::stream& stream,
                dnnl::engine& eng,
                unordered_map<string, dnnl::memory>& inputs,
                grp::Graph& g);
        void Execute();
    private:
        inline dnnl::memory& GetInput(string key) {
            auto srcMemoryPair = inputs.find(key);
            if(srcMemoryPair == inputs.end()) {
                throw std::runtime_error("source " + key + " not found");
            }
            return srcMemoryPair->second;
        }
        void InitNode(std::shared_ptr<node::Node> n);
        void InitConvNode(std::shared_ptr<node::ConvNode> n);
        void InitBnNode(std::shared_ptr<node::BatchNormNode> n);
        void InitShapeNode(std::shared_ptr<node::ShapeNode> n);
        void InitGatherNode(std::shared_ptr<node::GatherNode> n);
    };
}

#endif //SERVER_ENGINE_H
