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

//namespace node {
//    class Node;
//}

using namespace std;

namespace grp {

    class Graph {
    private:
        node::Node * root;
        vector<node::Node> nodes; // in topo-sort order
        map<string, vector<int>> inputDims;
        map<string, vector<int>> outputDims;
        unordered_map<string, ten::Tensor> weights;
    public:
        Graph();
        Graph(map<string, vector<int>> inDims, map<string, vector<int>> outDims, unordered_map<string, ten::Tensor> w);
        ~Graph();
        void SetRoot(node::Node* root);
        void AddNode(node::Node node);
        vector<node::Node> GetNodes();
        const unordered_map<string, ten::Tensor> GetWeights();
        void Forward();
    };

}

#endif //SERVER_GRAPH_H
