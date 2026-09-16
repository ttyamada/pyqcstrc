#include <iostream>
#include <vector>
#include <stdexcept>
#include "crsys.hpp"
#include "qnnum.hpp"
#include "qnndarray.hpp"

using namespace std;

void printqnv(const string& str, const QnNdarray& qnv) {
    auto shape = qnv.shape();
    int ndim = shape.size();

    if (ndim == 1) {  // for qnvector
        cout << str << " [ ";
        for (int i = 0; i < shape[0]; ++i) {
            cout << qn2npa(qnv[i]) << " ";
        }
        cout << "]" << endl;
    } else if (ndim == 2) { // for triangle/tetrahedron or qnmatrix
        for (int i = 0; i < shape[0]; ++i) {
            cout << str << " [ ";
            for (int j = 0; j < shape[1]; ++j) {
                cout << qn2npa(qnv[i][j]) << " ";
            }
            cout << "]" << endl;
        }
    } else if (ndim == 3) { // for triangles/tetrahedra
        for (int i = 0; i < shape[0]; ++i) {
            cout << str << " [ ";
            for (int j = 0; j < shape[1]; ++j) {
                for (int k = 0; k < shape[2]; ++k) {
                    cout << qn2npa(qnv[i][j][k]) << " ";
                }
                cout << "]" << endl;
            }
        }
        cout << endl;
    } else {
        cerr << "ndim should be 1 2 or 3 but " << ndim << endl;
        exit(EXIT_FAILURE);
    }
}

void printqnv2(const string& str, const Qnvec& qnv1, const Qnvec& qnv2) {
    cout << str << " [ ";
    int n1 = qnv1.shape();
    for (int i = 0; i < n1; ++i) {
        auto j = qnv1[i];
        cout << qn2npa(j) << " ";
    }
    cout << "] [ ";
    int n2 = qnv2.shape();
    for (int i = 0; i < n2; ++i) {
        auto j = qnv2[i];
        cout << qn2npa(j) << " ";
    }
    cout << "]" << endl;
}

void printqnvs(const string& str, const Qnvec& qnv1) {
    auto shape = qnv1.shape();
    int ndim = shape.size();
    int n = shape[0];
    cout << str << endl;
    for (int j = 0; j < n; ++j) {
        printqnv("", qnv1[j]);
    }
}

bool eq(const Qnvec& qnv1, const Qnvec& qnv2) {
    if (qnv1.shape().size() != qnv2.shape().size()) {
        return false;
    }
    int n_ = qnv1.shape()[0];
    for (int i = 0; i < n_; ++i) {
        if (qnv1[i] != qnv2[i]) {
            return false;
        }
    }
    return true;
}

bool not_eq(const Qnvec& qnv1, const Qnvec& qnv2) {
    int n_ = qnv1.shape()[0];
    for (int i = 0; i < n_; ++i) {
        if (qnv1[i] != qnv2[i]) {
            return true;
        }
    }
    return false;
}
