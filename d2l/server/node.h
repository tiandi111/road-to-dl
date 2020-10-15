//
// Created by 田地 on 2020/9/22.
//

#ifndef SERVER_NODE_H
#define SERVER_NODE_H

#include <vector>
#include <string>

using std::vector;
using std::string;

namespace grp {
    class Graph;
}

namespace node {

    enum OpType {
        conv,
        bn,
        relu
    };

    class Node {
    private:
        vector<string> inputs;
        vector<string> outputs;
//        vector<Node&> succs;  // todo: this need to be done with shared pointers
        OpType type;
        std::shared_ptr<grp::Graph> g;
        int id;
    public:
        Node();
        Node(OpType t,
                int id,
                vector<string> inputs,
                vector<string> outputs,
                std::shared_ptr<grp::Graph> g);
        virtual ~Node() = default;
        const OpType Type() const;
        const vector<string> Inputs() const;
        const vector<string> Outputs() const;
        const std::shared_ptr<grp::Graph> GetGraph() const;
        int ID() const;
        virtual void Absorb(Node another);
    };

    class ConvNode : public Node {
    private:
        int group;
        vector<int> dilations;
        vector<int> kernelShape;
        vector<int> pads; // l, r, u, d
        vector<int> strides;
        vector<int> weightDims;
        vector<int> biasDims;
        string srcInputName;
        string weightName;
        string biasName;
        bool relu;
    public:
        ConvNode();
        virtual ~ConvNode() = default;
        ConvNode(
                OpType t,
                int id,
                vector<string> inputs,
                vector<string> outputs,
                std::shared_ptr<grp::Graph> g,
                int group,
                vector<int> dilations,
                vector<int> kernelShape,
                vector<int> pads,
                vector<int> strides,
                vector<int> weightDims,
                vector<int> biasDims,
                string srcName,
                string weightName,
                string biasName);
        const vector<int>& KernelShape() const;
        const vector<int>& Pads() const;
        const vector<int>& Strides() const;
        const vector<int>& WeightDims() const;
        const vector<int>& BiasDims() const;
        const string SrcInputName() const;
        const string WeightName() const;
        const string BiasName() const;
        const string OutputName() const;
        virtual void Absorb(std::shared_ptr<Node> another);
        void EnablePostRelu();
    };

    class BatchNormNode : public Node {
    private:
        float epsilon;
        float momentum; // factor used to compute running mean and standard
        vector<int> dim;
        string wightName;
        string biasName;
        string meanName;
        string varName;
        string srcInputName;
    public:
        BatchNormNode();
        virtual ~BatchNormNode() = default;
        BatchNormNode(
                node::OpType t,
                int id,
                const vector<string> inputs,
                const vector<string> outputs,
                std::shared_ptr<grp::Graph> g,
                float epsilon,
                float momentum,
                vector<int> dim,
                string wightName,
                string biasName,
                string meanName,
                string varName,
                string srcInputName);
        float Epsilon() const;
        float Momentum() const; // factor used to compute running mean and standard
        const vector<int>& Dim() const;
        string WightName() const;
        string BiasName() const;
        string MeanName() const;
        string VarName() const;
        string SrcInputName() const;
        string OutputName() const;
        virtual void Absorb(std::shared_ptr<Node> another);
    };

}

#endif //SERVER_NODE_H
