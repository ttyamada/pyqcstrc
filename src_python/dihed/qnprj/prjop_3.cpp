#include <iostream>
#include <vector>
#include <array>
#include "crsys.h"
#include "qnnum.h"
#include "qnvec.h"
#include "qnmat.h"
#include "qnmath.h"
#include "qnndarray.h"

using namespace std;

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

