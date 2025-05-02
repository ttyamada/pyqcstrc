//#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <iostream>
#include <chrono>

#include "crsys.hpp"
#include "qnnum.hpp"


using namespace std;
using namespace qnnum;

int main() {
    int isys = 4; // for octagonal
    crsys::crsys_init(isys);
    init(); // qnnum init
    cout << "N " << N << endl;
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    Qnnum qnn0 = zero();
    printqnn("qnn0", qnn0);
    Qnnum qnn1 = one();
    printqnn("qnn1", qnn1);
    Qnnum qnn2 = qnn1;
    int d1[] = {2, 0, 1};
    Qnnum qnn3(d1);
    printqnn("qnn2", qnn2);
    printqnn("qnn3", qnn3);
    
    if (qnn1 == qnn2) {
        cout << "qnn1==qnn2" << endl;
    }
    cout << "qnn1==qnn2: " << (qnn1 == qnn2) << endl;
    cout << "qnn0==qnn1: " << (qnn0 == qnn1) << endl;
    cout << "qnn0>qnn1: " << (qnn0 > qnn1) << endl;
    cout << "qnn0<qnn1: " << (qnn0 < qnn1) << endl;
    int d2[] = {1, 1, 2};
    Qnnum qnn4 = any(d2);
    printqnn("qnn4", qnn4);
    cout << "qnn4>qnn1: " << (qnn4 > qnn1) << endl;
    cout << "qnn4<qnn1: " << (qnn4 < qnn1) << endl;
    cout << "qnn4==qnn1: " << (qnn4 == qnn1) << endl;
    cout << endl;

    Qnnum qnn5 = -qnn4;
    Qnnum qnn6 = qnn1 + qnn2;
    Qnnum qnn7 = qnn1 - qnn2;
    Qnnum qnn8 = qnn1 * qnn3;
    Qnnum qnn9 = qnn1 / qnn3;
    printqnn("-qnn4", qnn5);
    printqnn("qnn1+qnn2", qnn6);
    printqnn("qnn1-qnn2", qnn7);
    printqnn("qnn1*qnn3", qnn8);
    printqnn("qnn1/qnn3", qnn9);
    int d3[] = {1, 1, 2};
    qnn5 = any(d3);
    qnn6 = qnn1 / qnn5;
    qnn7 = qnn5 / qnn5;
    qnn8 = qnn5 * 2;
    qnn9 = qnn5 / 2;
    printqnn("qnn5", qnn5);
    printqnn("qnn1/qnn5",qnn6);
    printqnn("qnn5/qnn5",qnn7);
    printqnn("qnn5*2", qnn8);
    printqnn("qnn5/2", qnn9);

    Qnnum qnt1=zero(); Qnnum qnt2=zero(); Qnnum qnt3=zero(); Qnnum qnt4=zero();
    for (int i=0; i<1000000; ++i) {
        qnt1 = qnn1+qnn2;
        qnt2 = qnn1-qnn2;
        qnt3 = qnn1*qnn3;
        qnt4 = qnn1/qnn3;
    }

    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    cout << "elapsed time = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
    cout << "elapsed time = " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[µs]" << endl;
    //cout << "elapsed time = " << chrono::duration_cast<chrono::nanoseconds> (end - begin).count() << "[ns]" << endl;
    return 0;
}
