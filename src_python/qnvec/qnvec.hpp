#include <iostream>
#include <vector>
#include <type_traits>
#include <stdexcept>
#include <array>
#include <cmath>

#include "crsys.hpp"
#include "qnnum.hpp"
//#include "qnndarray.hpp"

// this should be replaced by point (array<Qnnum>[2])
// this hould be independent of qnvec and qnndarray
// use geometry instead (point edge circle triangle tetrahedron)
// operators depend on each type (point, edge, circle, triangle, tetrahedron)
// for point, many operations are defined, while for others
// (squared) length (for edge, circle), == (for all) area (for triangle),
//  volume (for tetrahedron) are defined

// template will be useful for double and qnnum (vector.hpp)

class Qnvec : public QnNdarray {
public:
    static std::vector<int64_t> shape;

    Qnvec(int64_t n) {
        shape = {n};
        this->resize(shape[0]);
        Qnnum qn0 = zero(); // int2qnn(0, N)
        for (size_t i = 0; i < shape[0]; ++i) {
            (*this)[i] = qn0;
        }
    }

    Qnvec(const Qnvec&) = default;
    Qnvec& operator=(const Qnvec&) = default;

    Qnvec operator+(const Qnvec& b) const {
        return add(*this, b);
    }

    Qnvec operator-(const Qnvec& b) const {
        return sub(*this, b);
    }

    Qnvec& operator+=(const Qnvec& b) {
        return iadd(*this, b);
    }

    Qnvec& operator-=(const Qnvec& b) {
        return isub(*this, b);
    }

    Qnvec operator*(const auto& b) const {
        if constexpr (std::is_same_v<decltype(b), int>) {
            return mul_vector_i(*this, b);
        } else if constexpr (std::is_same_v<decltype(b), Qnnum>) {
            return mul_vector_qn(*this, b);
        }
    }

    Qnvec operator/(const int64_t b) const {
        return div_vector_i(*this, b);
    }

    bool operator==(const Qnvec& b) const {
        return eq(*this, b);
    }

    bool operator!=(const Qnvec& b) const {
        return not_eq(*this, b);
    }

    Qnvec copy() const {
        return *this;
    }
};

std::vector<int64_t> Qnvec::shape;

void qnvec_init() {
    global_n = crs::n;
    global_N = crs::N;
    if (crs::isys == 2) {
        n_e = 3; n_i = 3;
    } else {
        n_e = 2; n_i = 2;
    }
}

QnNdarray zerovs(const std::vector<int64_t>& shape) {
    int64_t nv = shape[0];
    int64_t n = shape[1];
    QnNdarray qnvs = qna::zeros(shape);
    for (int64_t i = 0; i < nv; ++i) {
        qnvs[i] = zerov(n);
    }
    return qnvs;
}

Qnvec zerov(int64_t n) {
    return Qnvec(n);
}

QnNdarray anyv(const NDArray<Qnnum>& v) {
    auto shape = v.shape();
    int64_t n = shape[0];
    Qnvec v1(n);
    for (int64_t i = 0; i < n; ++i) {
        v1[i] = v[i];
    }
    return v1;
}



class Qnvec {
public:
    std::vector<double> data;

    Qnvec(size_t size) : data(size) {}

    size_t shape() const {
        return data.size();
    }

    double& operator[](size_t index) {
        return data[index];
    }

    const double& operator[](size_t index) const {
        return data[index];
    }

    int ndim() const {
        return 1; // Assuming 1D for this example
    }
};

namespace qna {
    Qnvec zeros(const std::vector<size_t>& shape) {
        return Qnvec(shape[0]);
    }
}

class Qnnum {};

Qnvec copy(const Qnvec& v) {
    Qnvec v1(v.shape());
    for (size_t i = 0; i < v.shape(); ++i) {
        v1[i] = v[i];
    }
    return v1;
}

Qnvec add(const Qnvec& v1, const Qnvec& v2) {
    size_t n = v1.shape();
    Qnvec a(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = v1[i] + v2[i];
    }
    return a;
}

Qnvec& iadd(Qnvec& self, const Qnvec& b) {
    self = add(self, b);
    return self;
}

Qnvec sub(const Qnvec& v1, const Qnvec& v2) {
    size_t n = v1.shape();
    Qnvec a(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = v1[i] - v2[i];
    }
    return a;
}

Qnvec& isub(Qnvec& self, const Qnvec& b) {
    self = sub(self, b);
    return self;
}

Qnvec sub_vectors_qn(const std::vector<std::vector<double>>& v, const Qnvec& v0) {
    std::vector<size_t> shape = {v.size(), v[0].size()};
    Qnvec v1 = qna::zeros(shape);
    size_t ndim = shape.size();
    if (ndim == 2) {
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                v1[i] = v[i][j] - v0[j];
            }
        }
    }
    return v1;
}

Qnvec mul_vector_i(const Qnvec& v, int coeff) {
    if (v.ndim() == 1) {
        Qnvec a(v.shape());
        for (size_t i = 0; i < v.shape(); ++i) {
            a[i] = v[i] * coeff;
        }
        return a;
    } else {
        throw std::invalid_argument("incorrect shape in mul_vector_i");
    }
}

Qnvec mul_vector_qn(const Qnvec& v, const Qnnum& coeff) {
    if (v.ndim() == 1) {
        Qnvec a(v.shape());
        for (size_t i = 0; i < v.shape(); ++i) {
            a[i] = v[i] * coeff; // Assuming operator* is defined for Qnvec and Qnnum
        }
        return a;
    } else {
        throw std::invalid_argument("incorrect shape in mul_vector_qn");
    }
}

std::vector<std::vector<double>> mul_vectors_i(const std::vector<std::vector<double>>& vs, int coeff) {
    std::vector<size_t> shape = {vs.size(), vs[0].size()};
    if (shape.size() == 2) {
        std::vector<std::vector<double>> a(shape[0], std::vector<double>(shape[1]));
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                a[i][j] = vs[i][j] * coeff;
            }
        }
        return a;
    } else {
        throw std::invalid_argument("incorrect shape in mul_vectors_i");
    }
}

std::vector<Qnvec> mul_vectors_qn(const std::vector<Qnvec>& vs, const Qnnum& coeff) {
    if (vs[0].ndim() == 2) {
        std::vector<Qnvec> a(vs.size(), Qnvec(vs[0].shape()));
        for (size_t i = 0; i < vs.size(); ++i) {
            for (size_t j = 0; j < vs[i].shape(); ++j) {
                a[i][j] = vs[i][j] * coeff; // Assuming operator* is defined for Qnvec and Qnnum
            }
        }
        return a;
    } else {
        throw std::invalid_argument("incorrect shape in mul_vectors_qn");
    }
}

Qnvec div_vector_i(const Qnvec& v, int coeff) {
    if (v.ndim() == 1) {
        Qnvec a(v.shape());
        for (size_t i = 0; i < v.shape(); ++i) {
            a[i] = v[i] / coeff;
        }
        return a;
    } else {
        throw std::invalid_argument("incorrect shape in div_vector_i");
    }
}


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
