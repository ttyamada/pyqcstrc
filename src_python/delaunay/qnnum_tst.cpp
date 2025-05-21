//#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <iostream>
#include <chrono>
#include <iomanip>

#include "crsys.hpp"
#include "qnnum.hpp"

using namespace std;
using namespace qnnum;

int main() {
    int isys = 4; // for octagonal
    crsys::crsys_init(isys);
    qnnum_init(); // qnnum init
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

    cout << std::boolalpha;  // for boolian true and false 
    if (qnn1 == qnn2) {
        cout << "qnn1==qnn2" << endl;
    }
    bool b0 = (qnn1 == qnn2);
    bool b1 = (qnn0 == qnn1);
    bool b2 = (qnn0 > qnn1);
    bool b3 = (qnn0 < qnn1);
    cout << "qnn1==qnn2: " << b0 << endl;
    cout << "qnn0==qnn1: " << b1 << endl;
    cout << "qnn0>qnn1: " << b2 << endl;
    cout << "qnn0<qnn1: " << b3 << endl;
    int d2[] = {1, 1, 2};
    Qnnum qnn4 = any(d2);
    printqnn("qnn4", qnn4);
    bool b4 = (qnn4 > qnn1);
    bool b5 = (qnn4 < qnn1);
    bool b6 = (qnn4 == qnn1);
    cout << "qnn4>qnn1: " << b4 << endl;
    cout << "qnn4<qnn1: " << b5 << endl;
    cout << "qnn4==qnn1: " << b6 << endl;
    cout << "qnn4>=qnn1: " << (qnn4>=qnn1) << endl;
    cout << "qnn4<=qnn1: " << (qnn4<=qnn1) << endl;
    cout << "qnn4>=qnn4: " << (qnn4>=qnn4) << endl;
    cout << "qnn4<=qnn4: " << (qnn4<=qnn4) << endl;
    cout << endl;

    Qnnum qnn5 = -qnn4;
    Qnnum qnn6 = qnn1 + qnn2;
    Qnnum qnn7 = qnn1 - qnn2;
    Qnnum qnn8 = qnn1 * qnn3;
    Qnnum qnn9 = qnn1 / qnn3;
    printqnn("qnn5=-qnn4", qnn5);
    printqnn("qnn1+qnn2", qnn6);
    printqnn("qnn1-qnn2", qnn7);
    printqnn("qnn1*qnn3", qnn8);
    printqnn("qnn1/qnn3", qnn9);
    int d3[] = {1, 1, 2};
    Qnnum qnn10 = any(d3);
    Qnnum qnn11 = qnn1 / qnn5;
    Qnnum qnn12 = qnn5 / qnn5;
    Qnnum qnn13 = qnn5 * 2;
    Qnnum qnn14 = qnn5 / 2;
    printqnn("qnn10", qnn10);
    printqnn("qnn1/qnn5",qnn11);
    printqnn("qnn5/qnn5",qnn12);
    printqnn("qnn5*2", qnn13);
    printqnn("qnn5/2", qnn14);
    Qnnum qnn15 = qnn5;
    qnn15+=qnn5;
    printqnn("qnn5+=qnn5", qnn15);
    Qnnum qnn16 = qnn5;
    qnn16-=qnn5;
    printqnn("qnn5-=qnn5", qnn16);
    Qnnum qnn17 = qnn5;
    qnn17 *= qnn5;
    printqnn("qnn5*=qnn5", qnn17);
    Qnnum qnn18 = qnn5;
    qnn18 /= qnn5;
    printqnn("qnn5/=qnn5", qnn18);

    Qnnum infp = inf();
    Qnnum infn = -inf();
    Qnnum nan0 = nan();
    printqnn("inf",infp);
    printqnn("-inf",infn);
    printqnn("nan",nan0);
    Qnnum absinfn=abs(infn);
    printqnn("abs(-inf)",absinfn);
    cout<<"inf<qnn1 "<<(infp<qnn1)<<endl;
    cout<<"inf>qnn1 "<<(infp>qnn1)<<endl;
    cout<<"-inf<qnn1 "<<(infn<qnn1)<<endl;
    cout<<"-inf>qnn1 "<<(infn>qnn1)<<endl;

    Qnnum qnt1=zero(); Qnnum qnt2=zero(); Qnnum qnt3=zero(); Qnnum qnt4=zero();
    for (int i=0; i<100000000; ++i) {
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
