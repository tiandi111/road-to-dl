//
// Created by 田地 on 2020/9/22.
//

#ifndef SERVER_GRAPH_H
#define SERVER_GRAPH_H

#include "tensor.h"
#include "node.h"
#include <vector>
#include <map>
#include <string>
#include <unordered_map>
#include <memory>

//namespace node {
//    class Node;
//}

using namespace std;

namespace grp {

    class Graph {
    private:
        vector<std::shared_ptr<node::Node>> nodes; // in topo-sort order
        map<string, vector<int>> inputDims;
        map<string, vector<int>> outputDims;
        unordered_map<string, ten::Tensor> weights;
        unordered_map<string, vector<std::shared_ptr<node::Node>>> inputDepens;
    public:
        Graph();
        Graph(map<string, vector<int>> inDims, map<string, vector<int>> outDims, unordered_map<string, ten::Tensor> w);
        ~Graph() = default;
        void AddNode(std::shared_ptr<node::Node> node);
        const vector<std::shared_ptr<node::Node>>& GetNodes() const;
        const unordered_map<string, ten::Tensor>& GetWeights() const;
        unordered_map<string, ten::Tensor>& GetMutableWeights();
        const void * GetWeightHandle(string name) const;
        void Fuse();
    };
}

#endif //SERVER_GRAPH_H
