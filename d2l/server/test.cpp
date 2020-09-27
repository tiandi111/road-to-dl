//
// Created by 田地 on 2020/9/19.
//

//#include <vector>
#include <iostream>
#include <map>
//#include "graph.h"
using namespace std;

class A {
private:
    map<int, int> m;
public:
    A();
    map<int, int>& ReturnMap();
    void Print();
};

A::A() {}

map<int, int>& A::ReturnMap() {
    return this->m;
}

void ModifyMap(map<int, int> m);

void ModifyMap(map<int, int> m) {
    m.insert(std::pair<int,int>(1,100));
}

void A::Print() {
    cout<< this->m.size() <<endl;
    for (map<int, int>::const_iterator iter = this->m.begin(); iter != this->m.end(); iter++) {
        cout<< iter->first << iter->second <<endl;
    }
}

class B {
private:
    map<int, int> m;
public:
    B();
    map<int, int>& ReturnMap();
};

map<int, int>& B::ReturnMap() {
    return this->m;
}


int main() {
    A a = A();
    map<int, int>& amap = a.ReturnMap();
//    amap.insert(std::pair<int,int>(1,100));
    ModifyMap(amap);
//    a.Print();
    for (map<int, int>::const_iterator iter = amap.begin(); iter != amap.end(); iter++) {
        cout<< iter->first << iter->second <<endl;
    }
//    a.Print();
}

