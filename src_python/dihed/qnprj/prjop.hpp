#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include "crsys.hpp"
#include "qnnum.hpp"
#include "qnvec.hpp"
#include "qnmat.hpp"
#include "qnmath.hpp"
#include "qnndarray.hpp"

class Qnprj_Octa : public Qnmat {
public:
    static int n;

    Qnprj_Octa() : Qnmat(n, n) {
        Qnnum M0 = qnn::any({0, 0, 1}); // 0
        Qnnum M1 = qnn::any({1, 0, 1}); // 1
        Qnnum M2 = qnn::any({-1, 0, 1}); // -1
        Qnnum M3 = qnn::any({0, 1, 2}); // sqrt(2)/2 t1
        Qnnum M4 = qnn::any({0, -1, 2}); // -sqrt(2)/2 t2=-t1

        std::vector<std::vector<Qnnum>> prj0 = {
            {M1, M0, M1, M0, M0},
            {M3, M3, M4, M3, M0},
            {M0, M1, M0, M2, M0},
            {M4, M3, M3, M3, M0},
            {M0, M0, M0, M0, M1}
        };

        this->prj0 = prj0;
        this->prji = qmt::qnmatinv(prj0, n); // get inversion matrix of prj
        this->n = n;
        this->N = crs::N;
        this->shape = {n, n};
        this->scl = 1.0;
        this->scly = 1.0;

        auto prj0f = qnm::qnm2flt(prj0); // for float number
        auto prjif = qmt::matinv_f(prj0f, n);
        this->prj0f = prj0f;
        this->prjif = prjif;
    }
};

class Qnprj_Deca : public Qnmat {
public:
    static int n;

    Qnprj_Deca() : Qnmat(n, n) {
        Qnnum M0 = qnn::any({0, 0, 1}); // 0
        Qnnum M1 = qnn::any({1, 1, 4}); // tau/2=c2
        Qnnum M2 = qnn::any({-1, 1, 4}); // tau^{-1}/2
        Qnnum M3 = qnn::any({1, 0, 1}); // 1
        Qnnum tau = qnn::any({1, 1, 2});

        Qnnum M5 = M2; // tau^{-1}/2=c1
        Qnnum M6 = -M1; // -tau/2=c2
        Qnnum M7 = M2 * 2; // s2/(s1)=tau^{-1}
        Qnnum M8 = -M7; // -s2/(s1)=-tau^{-1}
        Qnnum M9 = M3; // s1/(s1)=1
        Qnnum M10 = -M9; // -s1/(s1))=-1

        std::vector<std::vector<Qnnum>> prj0 = {
            {M5, M9, M6, M7, M0},
            {M6, M7, M5, M10, M0},
            {M6, M8, M5, M9, M0},
            {M5, M10, M6, M8, M0},
            {M0, M0, M0, M0, M3}
        };

        this->prj0 = prj0;
        this->prji = qmt::qnmatinv(prj0, n); // get inversion matrix of prj
        this->n = n;
        this->N = crs::N;
        this->shape = {n, n};
        this->scl = 2.0 / std::sqrt(5.0);
        this->scly = crs::scly; // s1
        this->scly2 = (M3 - M5 * M5); // (1-c1^2)

        auto prj0f = qnm::qnm2flt(prj0); // for float number
        auto prjif = qmt::matinv_f(prj0f, n);
        this->prj0f = prj0f;
        this->prjif = prjif;
    }
};

class Qnprj_Dode : public Qnmat {
public:
    static int n;

    Qnprj_Dode() : Qnmat(n, n) {
        Qnnum M0 = qnn::any({0, 0, 1}); // 0
        Qnnum M1 = qnn::any({1, 0, 1}); // 1
        Qnnum M2 = -M1;                 // -1
        Qnnum M3 = qnn::any({1, 0, 2}); // 1/2
        Qnnum M4 = -M3;                 // -1/2
        Qnnum M5 = qnn::any({0, 1, 2}); // sqrt(3)/2
        Qnnum M6 = -M5;                 // -sqrt(3)/2

        std::vector<std::vector<Qnnum>> prj0 = {
            {M1, M0, M1, M0, M0},
            {M5, M3, M6, M3, M0},
            {M3, M5, M3, M6, M0},
            {M0, M1, M0, M1, M0},
            {M0, M0, M0, M0, M1}
        };

        this->prj0 = prj0;
        this->prji = qmt::qnmatinv(prj0, n); // get inversion matrix of prj
        this->n = n;
        this->N = crs::N;
        this->shape = {n, n};
        this->scl = 2.0 / std::sqrt(6.0);  // for vesta or qnn2flt
        this->scly = 1.0;

        auto prj0f = qnm::qnm2flt(prj0);  // for float number
        auto prjif = qmt::matinv_f(prj0f, n);
        this->prj0f = prj0f;
        this->prjif = prjif;
    }
};

class Qnprj_Icos : public Qnmat {
public:
    static int n;

    Qnprj_Icos() : Qnmat(n, n) {
        Qnnum M0 = qnn::any({0, 0, 1}); // 0
        Qnnum M1 = qnn::any({1, 0, 1}); // 1
        Qnnum M2 = -M1;                 // -1
        Qnnum M3 = qnn::any({1, 1, 2}); // tau=(1+sqrt(5))/2
        Qnnum M4 = -M3;                 // -tau

        std::vector<std::vector<Qnnum>> prj0 = {
            {M1, M3, M0, M3, M2, M0},
            {M3, M0, M1, M2, M0, M3},
            {M3, M0, M2, M2, M0, M4},
            {M0, M1, M4, M0, M3, M1},
            {M2, M3, M0, M4, M2, M0},
            {M0, M1, M3, M0, M3, M2}
        };

        this->prj0 = prj0;
        this->prji = qmt::qnmatinv(prj0, n); // get inversion matrix of prj
        this->n = n;
        this->N = crs::N;
        this->shape = {n, n};
        double tau = (1.0 + std::sqrt(5.0)) / 2.0;
        this->scl = 1.0 / std::sqrt(2.0 + tau);
        this->scly = 1.0;

        auto prj0f = qnm::qnm2flt(prj0);  // for float number
        auto prjif = qmt::matinv_f(prj0f, n);
        this->prj0f = prj0f;
        this->prjif = prjif;
    }
};

int isys, n, N;
vector<vector<double>> prj0, prji, prj0t, prjit;
vector<vector<double>> prj0f, prjif;
double scl;

void prjop_init() {
    isys = crs::isys;
    n = crs::n;
    N = crs::N;
    Prjop();
}

auto Prjop() {
    if (isys == 2) { // projection operator for icosahedral
        auto prj = Qnprj_Icos();
    } else if (isys == 3) { // projection operator for decagonal
        auto prj = Qnprj_Deca();
    } else if (isys == 4) { // projection operator for octagonal
        auto prj = Qnprj_Octa();
    } else if (isys == 5) { // projection operator dodecagonal
        auto prj = Qnprj_Dode();
    }
    prj0 = prj.prj0;
    prji = prj.prji;
    prj0f = prj.prj0f;
    prjif = prj.prjif;
    prj0t = qmt::matrixtr(prj0); // transposed prj matrix
    prjit = qmt::matrixtr(prji); // transposed prji matrix 
    scl = prj.scl;
    return prj;
}

void tstwt_prjop() {
    qnm::printqnm("prj0", prj0);
    qnm::printqnm("prji", prji);
    qnm::printqnm("prj0t", prj0t);
    qnm::printqnm("prjit", prjit);
    auto unitm = qnm::zerom({n, n});
    unitm = qnm::matmul(prji, prj0);
    qnm::printqnm("unitm", unitm);
    printfm("prj0f", prj0f, n);
    auto prjif = qmt::matinv_f(prj0f, n);
    printfm("prjif", prjif, n);
    auto unitmf = qnm::matmul(prjif, prj0f);
    printfm("unitmf", unitmf, n);
}

auto copy(const vector<vector<double>>& qna1) {
    return qnm::copy(qna1);
}

auto prjvec(const vector<double>& v) -> vector<double> {
    int n = crs::n;
    vector<double> vei(n);
    copy(vei.begin(), vei.begin() + 3, prjvec_e(v).begin());
    if (isys > 2) {
        copy(vei.begin() + 3, vei.begin() + 5, prjvec_i(v).begin());
    } else {
        copy(vei.begin() + 3, vei.begin() + 6, prjvec_i(v).begin());
    }
    return vei;
}

auto prjvec_e(const vector<double>& v) -> vector<double> {
    auto vei = qnm::matmul(v, prj0);  // v assumed to be qnvec
    vector<double> ve(3);
    if (isys > 2) { // dihedral
        copy(vei.begin(), vei.begin() + 2, ve.begin());
        ve[2] = vei[4];
    } else if (isys == 2) { // icosahedral
        copy(vei.begin(), vei.begin() + 3, ve.begin());
    }
    return ve;
}

auto prjvec_i(const vector<double>& v) -> vector<double> {
    auto vei = qnm::matmul(v, prj0);
    int ni;
    vector<double> vi;
    if (isys > 2) { // dihedral
        ni = 2;
        vi.resize(ni);
        copy(vei.begin() + 2, vei.begin() + 4, vi.begin());
    } else if (isys == 2) { // icosahedral
        ni = 3;
        vi.resize(ni);
        copy(vei.begin() + 3, vei.begin() + 6, vi.begin());
    }
    return vi;
}

/ alias for prjop_i
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

