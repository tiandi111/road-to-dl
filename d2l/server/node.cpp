//
// Created by 田地 on 2020/9/22.
//

#include "node.h"
#include "graph.h"
#include <vector>
#include <iostream>

using namespace std;

node::Node::Node() {}

node::Node::Node(
        node::OpType t,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph* g) {
    if(!g) {
        throw "link to nil graph";
    }
//    this->inputDims = inDims;
//    this->outputDims = outDims;
    this->type = t;
    this->id = id;
    this->g = g;
    this->inputs = inputs;
    this->outputs = outputs;
}

node::Node::~Node() {}

node::OpType node::Node::Type() const {
    return this->type;
}

vector<string> node::Node::Inputs() const {
    return this->inputs;
}

vector<string> node::Node::Outputs() const {
    return this->outputs;
}

void node::Node::AddSucc(Node* succ) {
    if(!succ) {
        throw "add nil successor node";
    }
    this->succs.push_back(succ);
}

void node::Node::Forward() {
    cout<< (this->id) <<endl;
    for(int i=0; i<this->succs.size(); i++) {
        succs[i]->Forward();
    }
}

node::ConvNode::ConvNode() {}

node::ConvNode::ConvNode(
        node::OpType t,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph* g,
        int group,
        vector<int> dilations,
        vector<int> kernelShape,
        vector<int> pads,
        vector<int> strides) : Node(t, id, inputs, outputs, g) {
    this->group = group;
    this->dilations = dilations;
    this->kernelShape = kernelShape;
    this->pads = pads;
    this->strides = strides;
}

const vector<int>& node::ConvNode::KernelShape() const {
    return this->kernelShape;
}
const vector<int>& node::ConvNode::Pads() const {
    return this->pads;
}
const vector<int>& node::ConvNode::Strides() const {
    return this->strides;
}