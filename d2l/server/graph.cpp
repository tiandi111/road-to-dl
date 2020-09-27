//
// Created by 田地 on 2020/9/22.
//

#include "graph.h"
#include "node.h"

using namespace grp;

Graph::Graph() {}

Graph::Graph(map<string, vector<int>> inDims, map<string, vector<int>> outDims, unordered_map<string, ten::Tensor> w) {
    this->inputDims = inDims;
    this->outputDims = outDims;
    this->weights = w;
}

void Graph::SetRoot(node::Node* r) {
    this->root = r;
}

void Graph::AddNode(node::Node node) {
    this->nodes.push_back(node);
}

vector<node::Node> Graph::GetNodes() {
    return this->nodes;
}

const unordered_map<string, ten::Tensor> Graph::GetWeights() {
    return this->weights;
}

void Graph::Forward() {
    if(!this->root) {
        throw "nil root";
    }
    this->root->Forward();
}