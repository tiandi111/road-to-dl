//
// Created by 田地 on 2020/10/27.
//

#include "intergration_test.h"
#include "loader.h"
#include "context.h"
#include "engine.h"
#include "engine_test.h"
#include "tensor.h"
#include "test.h"
#include <fstream>

void interTest::TestInference() {
    ifstream in("/Users/tiandi03/road-to-dl/d2l/server/test/models/export.onnx", ios_base::binary);
    auto g = load::LoadOnnx(&in);
    eng::MKLEngine mklEngine("cpu", eng::DeviceType::cpu);

    vector<float> data = {
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
    };
    vector<int64_t> dims = {1, 1, 4, 4};
    ten::Tensor src(dims, ten::f32, (char*)data.data(), 64);
    ctx::InputContext inCtx({{"input.1", src}});
    ctx::OutputContext ouCtx({{"11", ten::Tensor({1, 3, 4, 4}, ten::f32)}});

    mklEngine.Execute(inCtx,ouCtx, g);

    auto got = ouCtx.Outputs().find("11");
    AssertFalse(got==ouCtx.Outputs().end(), "TestInference, case1, not exist");
    for(int i=0; i<48; i++) {
        float out = *((float*)got->second.Data().data() + i);
        cout<< out <<endl;
//        AssertEqual(out, float(1), "TestInference, case1, wrong output");
    }
}