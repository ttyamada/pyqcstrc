#include <iostream>
#include <vector>
#include <stdexcept>
#include "qnn.h" // Assuming these headers exist for Qnnum, Qnmat, Qnvec
#include "qnv.h"
#include "qna.h"

using namespace std;

Qnmat sub(const Qnmat& ma1, const Qnmat& ma2) {
    auto shape = ma1.shape();
    int n1 = ma1.ndim();
    int n2 = ma2.ndim();
    Qnmat a(shape);
    if (n1 == 1 && n2 == 1) { // vectors
        for (int i = 0; i < shape[0]; i++) {
            a[i] = ma1[i] - ma2[i];
        }
        return a;
    } else if (n1 == 2 && n2 == 2) { // matrices
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                a[i][j] = ma1[i][j] - ma2[i][j];
            }
        }
        return a;
    }
    throw invalid_argument("Invalid dimensions for subtraction");
}

Qnmat iadd(Qnmat& ma1, const Qnmat& ma2) {
    ma1 = add(ma1, ma2);
    return ma1;
}

Qnmat isub(Qnmat& ma1, const Qnmat& ma2) {
    ma1 = sub(ma1, ma2);
    return ma1;
}

Qnmat mul_scl(Qnmat& ma1, const Qnnum& scl) {
    int ndim = ma1.ndim();
    if (ndim == 1) {
        for (int i = 0; i < ma1.shape()[0]; i++) {
            ma1[i] = ma1[i] * scl;
        }
    } else if (ndim == 2) {
        for (int i = 0; i < ma1.shape()[0]; i++) {
            for (int j = 0; j < ma1.shape()[1]; j++) {
                ma1[i][j] = ma1[i][j] * scl;
            }
        }
    }
    return ma1;
}

Qnmat mul(const Qnmat& ma1, const Qnmat& ma2) {
    int ndm1 = ma1.ndim();
    int ndm2 = ma2.ndim();
    Qnmat ma3;

    if (ndm1 == 1 && ndm2 == 1) { // dot product
        if (ma1.shape()[0] == ma2.shape()[0]) {
            ma3 = qnn::zero();
            for (int i = 0; i < ma1.shape()[0]; i++) {
                ma3 += ma1[i] * ma2[i];
            }
        } else {
            throw invalid_argument("ma1.shape[0] should be equal to ma2.shape[0] for ndim1==ndim2");
        }
    } else if (ndm1 == 1 && ndm2 == 2) { // vec*matrix
        if (ma1.shape()[0] == ma2.shape()[0]) {
            ma3 = qnv::zerov(ma2.shape()[1]);
            for (int j = 0; j < ma2.shape()[1]; j++) {
                for (int i = 0; i < ma1.shape()[0]; i++) {
                    ma3[j] += ma1[i] * ma2[i][j];
                }
            }
        } else {
            throw invalid_argument("ma1.shape[0] should be equal to ma2.shape[0] for ndim1=1 and ndim2=2");
        }
    } else if (ndm1 == 2 && ndm2 == 1) { // matrix*vec
        if (ma1.shape()[1] == ma2.shape()[0]) {
            ma3 = qnv::zerov(ma1.shape()[0]);
            for (int j = 0; j < ma2.shape()[0]; j++) {
                for (int i = 0; i < ma1.shape()[0]; i++) {
                    ma3[j] += ma1[j][i] * ma2[i];
                }
            }
        } else {
            throw invalid_argument("ma1.shape[1] should be equal to ma2.shape[0] for ndim1=2 and ndim2=1");
        }
    } else if (ndm1 == 2 && ndm2 == 2) {
        if (ma1.shape()[0] == ma1.shape()[1] && ma2.shape()[0] == ma2.shape()[1] && ma1.shape()[0] == ma2.shape()[0]) {
            ma3 = qnv::zerov(ma1.shape());
            for (int k = 0; k < ma2.shape()[0]; k++) {
                for (int j = 0; j < ma1.shape()[0]; j++) {
                    for (int i = 0; i < ma1.shape()[0]; i++) {
                        ma3[k][j] += ma1[k][i] * ma2[i][j];
                    }
                }
            }
        } else {
            throw invalid_argument("square matrix is assumed for ndm1=ndm2=2");
        }
    }

    return ma3;
}

Qnmat pow(const Qnmat& ma, int n_) {
    auto shape = ma.shape();
    int mx = shape[0];
    int my = shape[1];
    if (mx == my) {
        if (n_ == 0) {
            return Qnmat::identity(mx);
        } else {
            Qnmat tmp = Qnmat::unitm(n_, N); // Assuming unitm is a static method
            for (int i = 0; i < n_; i++) {
                tmp = mul(tmp, ma);
            }
            return tmp;
        }
    } else {
        throw invalid_argument("matrix has not regular shape");
    }
}

