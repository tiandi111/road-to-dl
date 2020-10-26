//
// Created by 田地 on 2020/9/22.
//

#include "graph.h"
#include "node.h"
#include <iostream>
using namespace grp;

Graph::Graph() {}

Graph::Graph(map<string, vector<int>> inDims, map<string, vector<int>> outDims, unordered_map<string, ten::Tensor> w) {
    inputDims = inDims;
    outputDims = outDims;
    weights = w;
}

void Graph::AddNode(std::shared_ptr<node::Node> node) {
    this->nodes.push_back(node);
    for(const auto in : node->Inputs()) {
        this->inputDepends[in].push_back(node);
    }
}

const vector<std::shared_ptr<node::Node>>& Graph::GetNodes() const {
    return this->nodes;
}

const unordered_map<string, ten::Tensor>& Graph::GetWeights() const {
    return this->weights;
}

unordered_map<string, ten::Tensor>& Graph::GetMutableWeights(){
    return this->weights;
}

const void * Graph::GetWeightHandle(string name) const {
    auto got = this->weights.find(name);
    if (got == this->weights.end()) {
        return nullptr;
    }
    return static_cast<const void * >(got->second.Data().data());
}

const ten::Tensor& Graph::GetWeightTensor(string name) const {
    auto got = weights.find(name);
    if(got == weights.end()) {
        throw std::runtime_error("weight " + name + " not found");
    }
    return got->second;
}

void Graph::Fuse() {
    for(auto it = this->nodes.begin(); it<this->nodes.end(); it++) {
        if(it+1 == this->nodes.end()) {
            return;
        }
        auto node = *it;
        if(node->Type() == node::OpType::conv || node->Type() == node::OpType::bn) {
            string outputName = node->Outputs()[0];
            if(this->inputDepends[outputName].size() != 1) {
                it++;
                continue;
            }
            auto succNode = this->inputDepends[outputName][0];
            if(node->Type() == node::OpType::conv) {
                std::dynamic_pointer_cast<node::ConvNode>(node)->Absorb(succNode);
            }
            if(node->Type() == node::OpType::bn) {
                std::dynamic_pointer_cast<node::BatchNormNode>(node)->Absorb(succNode);
            }
            this->inputDepends.erase(outputName);
            this->nodes.erase(this->nodes.begin()+succNode->ID());
            break;
        }
    }
}

