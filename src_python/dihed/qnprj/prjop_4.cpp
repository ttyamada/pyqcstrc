#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include "crsys.h"
#include "qnnum.h"
#include "qnvec.h"
#include "qnmat.h"
#include "qnmath.h"
#include "qnndarray.h"

using namespace std;

// alias for prjop_i
Qnvec projection3(Qnvec v) {
    return prjvec_i(v);
}

// alias for prjop
Qnvec projection_numerical(Qnvec vn) {
    return prjvec(vn);
}

QnNdarray projection3_sets_numerical(QnNdarray vns) {
    // Parameters
    // vsn: qnvector array
    // set of nD vectors shape (num,n) assumed or
    // (num,3,n) or (num,4,n) for triangles or tetrahedra 
    auto shape = vns.shape();
    int ndim = shape.size();
    int isys = crs::isys;
    int ni = (isys > 2) ? 2 : 3;
    QnNdarray m;

    if (ndim == 1) {
        m = QnNdarray(ni);
        m = projection3(vns);
    } else if (ndim == 2) {
        m = QnNdarray(shape[0], ni);
        for (int i = 0; i < shape[0]; i++) {
            m[i] = projection3(vns[i]);
        }
    } else if (ndim == 3) {
        m = QnNdarray(shape[0], shape[1], ni);
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                m[i][j] = projection3(vns[i][j]);
            }
        }
    }
    return m;
}

// alias for prjop_e
Qnvec projection_numerical_par(Qnvec v) {
    return prjvec_e(v);
}

// alias for prjop_i
Qnvec get_internal_component_numerical(Qnvec v) {
    return prjvec_i(v);
}

void check_ltv(int n, int N) {
    // check lattice vector external and internal space components
    vector<Qnnum> ndv(27); // 3^3
    n = 5;
    Qnnum V0 = Qnnum({0, 0, 1});
    for (int i1 = -1; i1 <= 1; i1++) {
        for (int i2 = -1; i2 <= 1; i2++) {
            for (int i3 = -1; i3 <= 1; i3++) {
                for (int i4 = -1; i4 <= 1; i4++) {
                    Qnnum V1 = Qnnum({i1, 0, 1});
                    Qnnum V2 = Qnnum({i2, 0, 1});
                    Qnnum V3 = Qnnum({i3, 0, 1});
                    Qnnum V4 = Qnnum({i4, 0, 1});
                    // V0 = Qnnum(array({0, 0, 1}), N);
                    // VT = array({v1, v2, v3, V4, V0});
                    auto vt = Qnvec::anyv(n, N, {V1, V2, V3, V4, V0});
                    qnn::printqnv("ndv", vt); // print qnvector expression
                    auto qnv = prj0 * vt; // qnv = vt * prj0; // external internal components
                    qnn::printqnv("qnv", qnv); // print qnvector expression
                    n++;
                }
            }
        }
    }
}

void print_prj(const string& str, Qnmat prj) {
    // print(prj);
    cout << str << endl;
    int n = prj.shape()[0];
    int N = prj.N;
    cout << "n: " << n << ", N: " << N << endl;

    qnm::printqnm(str, prj);

    Qnmat b = qnm::copy(prj);
    cout << str << endl;
    qnm::printqnm("str", b);
}

void printfm(const string& str, const Qnmat& prj3f, int n) {
    cout << str << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << fixed << setprecision(8) << prj3f[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

vector<vector<double>> qnm2flnm(Qnmat prj) {
    int n = prj.shape()[0];
    vector<vector<double>> prj0f(n, vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            auto a = prj[i][j];
            // cout << "a: " << a.n[0] << ", " << a.n[1] << ", " << a.n[2] << endl; // for test
            prj0f[i][j] = qnn::qn2flt(a) * scl;
            // cout << "f: " << prj0f[i][j] << endl; // for test
        }
    }
    return prj0f;
}

