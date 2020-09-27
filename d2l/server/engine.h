//
// Created by 田地 on 2020/9/25.
//

#ifndef SERVER_ENGINE_H
#define SERVER_ENGINE_H

#include "dnnl.hpp"
#include "graph.h"
#include "input_context.h"
#include <string>
#include <vector>

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
        ~Engine();
//        Execute(ictx::InputContext ctx, grp::Graph g);
    };

    class MKLEngine : Engine {
    private:
        std::unordered_map<string, dnnl::primitive> prims;
        dnnl::engine eng;
    public:
        MKLEngine(string name, DeviceType t);
        ~MKLEngine();
        void Execute(ictx::InputContext ctx, grp::Graph* g);
    };

    class MKLExecutionContext {
    private:
        dnnl::stream stream;
        dnnl::engine eng;
        vector<dnnl::primitive> prims;
        vector<unordered_map<int, dnnl::memory>> args;
        unordered_map<string, dnnl::memory>& inputs;
        grp::Graph* g;
    public:
        MKLExecutionContext(
                dnnl::stream stream,
                dnnl::engine eng,
                unordered_map<string, dnnl::memory>& inputs,
                grp::Graph* g);
        void Execute();
    private:
        void InitNode(const node::Node& n);
        void InitConvNode(const node::ConvNode& n);
    };

}

#endif //SERVER_ENGINE_H
