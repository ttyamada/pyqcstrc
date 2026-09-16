#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include "crsys.hpp"
#include "qnnum.hpp"
#include "qnndarray.hpp"

using Qnvec = std::vector<Qnnum>; // Assuming Qnnum is a class defined in qnnum.h

// cros == outer_product (cross product) 
// n should be 3
Qnvec cros(const Qnvec& v1, const Qnvec& v2) {
    if (v1.size() != 3 || v2.size() != 3) {
        std::cout << "dimension should be 3 for cross product" << std::endl;
        exit(1);
    }
    auto a = v1[1] * v2[2]; 
    auto b = v1[2] * v2[1]; 
    auto c1 = a - b;

    a = v1[2] * v2[0]; 
    b = v1[0] * v2[2]; 
    auto c2 = a - b;

    a = v1[0] * v2[1]; 
    b = v1[1] * v2[0]; 
    auto c3 = a - b;

    return Qnvec{c1, c2, c3};
}

// for dihedral and icosahedral excluding decagonal
Qnnum dot(const Qnvec& v1, const Qnvec& v2) {
    int isys = crs::isys;
    if (isys == 3) {
        auto s2 = crs::scly; 
        return v1[0] * v2[0] + v1[1] * v2[1] * s2;
    } else {
        Qnnum v = qnn::zero(); 
        int n = v1.size();
        for (int i = 0; i < n; ++i) {
            v += (v1[i] * v2[i]);
        }
        return v;
    }
}

// equivalent to cross
Qnvec outer_product(const Qnvec& v1, const Qnvec& v2) {
    return cros(v1, v2);
}

// equivalent to dot
Qnnum inner_product(const Qnvec& v1, const Qnvec& v2) {
    return dot(v1, v2);
}

std::vector<std::array<int64_t, 3>> qnv2npa(const Qnvec& a) {
    int la = a.size();
    std::vector<std::array<int64_t, 3>> b(la);
    for (int i = 0; i < la; ++i) {
        auto ai = a[i];
        b[i] = {ai.n[0], ai.n[1], ai.n[2]};
    }
    return b;
}

std::vector<double> qnv2flt(const Qnvec& a) {
    int n = a.size();
    std::vector<double> b(n, 0.0);
    for (int i = 0; i < n; ++i) {
        auto ai = a[i];
        b[i] = (ai.n[0] + ai.n[1] * std::sqrt(N)) / ai.n[2];
    }
    auto scly = crs::scly;
    int isys = crs::isys;
    if (isys == 3) {
        b[1] *= scly;
        if (n > 3) {
            b[3] *= scly;
        }
    }
    return b;
}

Qnvec intv2qnv(const std::vector<int>& a) {
    int n = a.size();
    Qnnum qn0 = qnn::Qnnum({0, 0, 1}); 
    Qnvec b(n, qn0); 
    for (int i = 0; i < n; ++i) {
        b[i] = qnn::int2qnn(a[i]);
    }
    return b;
}
