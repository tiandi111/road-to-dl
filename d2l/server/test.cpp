//
// Created by 田地 on 2020/9/19.
//

//#include <vector>
#include <iostream>
#include <map>
#include <vector>
using namespace std;

int main() {
    float x[10];
    float y[10];
    float z[10];
    float *px = x;
    float *py = y;
    for(int i=0; i<10; i++) {
        cout<< px[i] * py[i] <<endl;
        cout<< px[i] <<endl;
        cout<< py[i] <<endl;
    }
}