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
        relu,
        gemm,
        shape,
        gather,
        mul,
        unsqueeze,
        concat,
        reshape
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

    // Note: we do not need relu node! The base node is enough!
    class ShapeNode : public Node {
    public:
        ShapeNode(OpType t,
                int id,
                vector<string> inputs,
                vector<string> outputs,
                std::shared_ptr<grp::Graph> g)
                : Node(t, id, inputs, outputs, g) {};
        inline string InputName() const {return this->Inputs()[0];}
        inline string OutputName() const {return this->Outputs()[0];}
    };

    // Gather data on the given dimension
    // e.g, data is 6×5×4×3, indices is 2×1, axis = 0, then output is 2×1×5×4×3
    // e.g, data is 6×5×4×3, indices is 2×1, axis = 1, then output is 6×2×1×4×3
    // e.g, data is 6×5×4×3, indices is 2×1, axis = 2, then output is 6×5×2×1×3
    // e.g, data is 6×5×4×3, indices is 2×1, axis = 0, then output is 6×5×4×2×1
    // Why this is usefule?
    // App1: when we flat matrix, we need to take each of the dimsions,
    // and multiply them all, gather can be used to take out a certain dim
    // App2: we may need to construct new features by combining original features in different way
    // todo: negative index
    class GatherNode : public Node {
    private:
        int axis;
    public:
        GatherNode(OpType t,
                  int id,
                  vector<string> inputs,
                  vector<string> outputs,
                  std::shared_ptr<grp::Graph> g,
                  int axis)
                : Node(t, id, inputs, outputs, g), axis(axis) {};
        inline int Axis() const {return this->axis;}
        inline string DataName() const {return this->Inputs()[0];}
        inline string IndicesName() const {return this->Inputs()[1];}
        inline string OutputName() const {return this->Outputs()[0];}
    };
}

#endif //SERVER_NODE_H
