#include <iostream>
#include <vector>
#include <cmath>
#include "crsys.h"
#include "qnnum.h"
#include "qnvec.h"
#include "qnmat.h"
#include "qnmath.h"
#include "qnndarray.h"

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

