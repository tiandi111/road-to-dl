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
        vector<string> inputs,
        vector<string> outputs,
        shared_ptr<grp::Graph> g) {
    this->type = t;
    this->id = id;
    this->inputs = inputs;
    this->outputs = outputs;
    this->g = g;
}

const node::OpType node::Node::Type() const {
    return this->type;
}

const vector<string> node::Node::Inputs() const {
    return this->inputs;
}

const vector<string> node::Node::Outputs() const {
    return this->outputs;
}

const std::shared_ptr<grp::Graph> node::Node::GetGraph() const {
    return this->g;
}

int node::Node::ID() const {
    return this->id;
}

void node::Node::Absorb(Node another) {
    throw std::runtime_error("node type " + to_string(this->Type()) + " does not support fuse");
}

node::ConvNode::ConvNode() {}

node::ConvNode::ConvNode(
        node::OpType t,
        int id,
        const vector<string> inputs,
        const vector<string> outputs,
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
        string biasName) : Node(t, id, inputs, outputs, g) {
    this->group = group;
    this->dilations = dilations;
    this->kernelShape = kernelShape;
    this->pads = pads;
    this->strides = strides;
    this->weightDims = weightDims;
    this->biasDims = biasDims;
    this->srcInputName = srcName;
    this->weightName = weightName;
    this->biasName = biasName;
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
const vector<int>& node::ConvNode::WeightDims() const {
    return this->weightDims;
}
const vector<int>& node::ConvNode::BiasDims() const {
    return this->biasDims;
}
const string node::ConvNode::SrcInputName() const {
    return this->srcInputName;
}
const string node::ConvNode::WeightName() const {
    return this->weightName;
}
const string node::ConvNode::BiasName() const {
    return this->biasName;
}
const string node::ConvNode::OutputName() const {
    return this->Outputs()[0];
}
void node::ConvNode::Absorb(std::shared_ptr<Node> another) {
    // todo: different data type
    if(another->Type() == node::bn) {
        std::shared_ptr<node::BatchNormNode> bnNode = std::dynamic_pointer_cast<node::BatchNormNode>(another);
        auto weights = this->GetGraph()->GetMutableWeights();
        auto & convWeight = weights[this->WeightName()];
        auto & convBias = weights[this->BiasName()];
        auto & mean = weights[bnNode->MeanName()];
        auto & var = weights[bnNode->VarName()];
        auto & bnWeight = weights[bnNode->WightName()];
        auto & bnBias = weights[bnNode->BiasName()];

        size_t OC = var.Data().size() >> 2;
        float * bvar = (float *) var.Data().data();
        float * wei  = (float *) bnWeight.Data().data();
        float * bias  = (float *) bnBias.Data().data();
        float * bmean = (float *) mean.Data().data();
        float scalar[OC];
        float shifter[OC];
        for(int i=0; i<OC; i++) {
            scalar[i] = wei[i] / bvar[i];
            shifter[i] = bias[i] - bmean[i] / bvar[i];
        }

        // scale weight
        size_t perOCSize = (convWeight.Data().size() >> 2) / OC;
        float * cw = (float *) convWeight.Data().data();
        for(size_t i=0; i<OC; i++) {
            size_t from = i * perOCSize;
            size_t to = from + perOCSize;
            for(; from<to; from++) {
                cw[from] *= scalar[i];
            }
        }

        // scale and shift bias
        float * cb = (float *) convBias.Data().data();
        for(size_t i=0; i<OC; i++) {
            cb[i] = scalar[i] * cb[i] + shifter[i];
        }

        return;
    }
    if(another->Type() == node::relu) {
        this->EnablePostRelu();
        return;
    }
    throw runtime_error("node type " + to_string(this->Type()) + " cannot absorb type " + to_string(another->Type()));
}
void node::ConvNode::EnablePostRelu() {
    this->relu = true;
}

node::BatchNormNode::BatchNormNode(){}
node::BatchNormNode::BatchNormNode(
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
        string srcInputName) :
        Node(t, id, inputs, outputs, g),
        epsilon(epsilon),
        momentum(momentum),
        dim(dim),
        wightName(wightName),
        biasName(biasName),
        meanName(meanName),
        varName(varName),
        srcInputName(srcInputName) {}

float node::BatchNormNode::Epsilon() const {
    return this->epsilon;
}
float node::BatchNormNode::Momentum() const {
    return this->momentum;
}
const vector<int>& node::BatchNormNode::Dim() const {
    return this->dim;
}
string node::BatchNormNode::WightName() const {
    return this->wightName;
}
string node::BatchNormNode::BiasName() const {
    return this->biasName;
}
string node::BatchNormNode::MeanName() const {
    return this->meanName;
}
string node::BatchNormNode::VarName() const {
    return this->varName;
}
string node::BatchNormNode::SrcInputName() const {
    return this->srcInputName;
}
string node::BatchNormNode::OutputName() const {
    return this->Outputs()[0];
}
void node::BatchNormNode::Absorb(std::shared_ptr<Node> another) {

}