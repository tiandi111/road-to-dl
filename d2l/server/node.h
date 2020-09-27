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
        vector<Node*> succs;
        OpType type;
        int id;
        grp::Graph* g;
    public:
        Node();
        Node(OpType t,
                int id,
                const vector<string>& inputs,
                const vector<string>& outputs,
                grp::Graph* g);
        ~Node();
        OpType Type() const;
        vector<string> Inputs() const;
        vector<string> Outputs() const;
        void AddSucc(Node* succ);
        void Forward();
    };

    class ConvNode : public Node {
    private:
        int group;
        vector<int> dilations;
        vector<int> kernelShape;
        vector<int> pads; // l, r, u, d
        vector<int> strides;
    public:
        ConvNode();
        ConvNode(
                OpType t,
                int id,
                const vector<string>& inputs,
                const vector<string>& outputs,
                grp::Graph* g,
                int group,
                vector<int> dilations,
                vector<int> kernelShape,
                vector<int> pads,
                vector<int> strides);
        const vector<int>& KernelShape() const;
        const vector<int>& Pads() const;
        const vector<int>& Strides() const;
    };
}

#endif //SERVER_NODE_H
