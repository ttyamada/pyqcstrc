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

