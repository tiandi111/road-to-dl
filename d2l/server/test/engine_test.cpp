//
// Created by 田地 on 2020/10/26.
//

#include "loader.h"
#include "context.h"
#include "engine.h"
#include "engine_test.h"
#include "tensor.h"
#include "test.h"
#include <fstream>

void engineTest::TestExecGemmNode() {
    ifstream in("/Users/tiandi03/road-to-dl/d2l/server/test/models/gemm.onnx", ios_base::binary);
    auto g = load::LoadOnnx(&in);
    eng::MKLEngine mklEngine("cpu", eng::DeviceType::cpu);

    vector<float> data = {2}; vector<int64_t> dims = {1, 1};
    ten::Tensor src(dims, ten::f32, (char*)data.data(), 4);
    ctx::InputContext inCtx({{"input", src}});
    ctx::OutputContext ouCtx({{"10", ten::Tensor({1}, ten::f32)}});

    mklEngine.Execute(inCtx,ouCtx, g);

    auto got = ouCtx.Outputs().find("10");
    AssertFalse(got==ouCtx.Outputs().end(), "TestExecGemmNode, case1, not exist");
    AssertEqual(*((float*)got->second.Data().data()), float(3), "TestExecGemmNode, case1, wrong output");
}
