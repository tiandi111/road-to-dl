//
// Created by 田地 on 2020/9/25.
//

#ifndef SERVER_INPUT_CONTEXT_H
#define SERVER_INPUT_CONTEXT_H

#include <unordered_map>
#include <string>
#include "tensor.h"

using namespace std;

namespace ictx {
    class InputContext {
    private:
        unordered_map<string, ten::Tensor> inputs;
    public:
        ~InputContext() = default;
        InputContext(unordered_map<string, ten::Tensor>& inputs);
        unordered_map<string, ten::Tensor>& Inputs();
    };
}


#endif //SERVER_INPUT_CONTEXT_H
