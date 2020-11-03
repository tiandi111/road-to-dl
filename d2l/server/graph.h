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
        // Returns the model's parameters
        unordered_map<string, ten::Tensor> weights;
        // Returns the edges of the computation graph, i.e which node depends on which input or other node's output
        // key: the input name
        // value: a vector of nodes that depends on the input
        unordered_map<string, vector<std::shared_ptr<node::Node>>> inputDepends;
    public:
        Graph() = default;
        Graph(map<string, vector<int>> inDims,
                map<string, vector<int>> outDims,
                unordered_map<string, ten::Tensor> w) : inputDims(inDims), outputDims(outDims), weights(w) {}

        ~Graph() = default;
        void AddNode(std::shared_ptr<node::Node> node);
        const vector<std::shared_ptr<node::Node>>& GetNodes() const;
        const unordered_map<string, ten::Tensor>& GetWeights() const;
        unordered_map<string, ten::Tensor>& GetMutableWeights();
        const void * GetWeightHandle(string name) const;
        const ten::Tensor& GetWeightTensor(string name) const;
        void Fuse();
    };
}

#endif //SERVER_GRAPH_H
