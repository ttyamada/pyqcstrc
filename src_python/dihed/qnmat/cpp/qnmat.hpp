#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <array>

#include "crsys.hpp"
#include "qnnum.hpp"
//#include "qnvec.hpp"
//#include "qnndarray.hpp"
#include "numeric.hpp"
#include "prjop.hpp"

// this should only depend on crsys and qnnum
// qnvec should be replaced by point, edge, triangle, tetrahedron
// defined in geometry

// template will be usefull for double and qnnum (matrix.hpp)

class Qnmat : public QnNdarray {
public:
    static Qnmat* create(const std::vector<int64_t>& shape) {
        return new Qnmat(shape);
    }

    Qnmat(const std::vector<int64_t>& shape) : QnNdarray(shape) {
        Qnnum qn0 = qnn::zero();
        this->N = N;
        this->shape = shape;
        int ndim = shape.size();

        if (ndim == 1) {
            for (int64_t i = 0; i < shape[0]; ++i) {
                (*this)[i] = qn0;
            }
        }

        if (ndim == 2) {
            for (int64_t i = 0; i < shape[0]; ++i) {
                for (int64_t j = 0; j < shape[1]; ++j) {
                    (*this)[i][j] = qn0;
                }
            }
        }
    }

    Qnmat operator+(const Qnmat& ma2) const {
        return add(*this, ma2);
    }

    Qnmat& operator+=(const Qnmat& ma2) {
        return iadd(*this, ma2);
    }

    Qnmat& operator-=(const Qnmat& ma2) {
        return isub(*this, ma2);
    }

    Qnmat operator@(const Qnmat& ma2) const {
        return mul(*this, ma2);
    }

    void set_mt(const std::vector<std::vector<Qnnum>>& mt) {
        auto shape = this->shape;
        for (int64_t i = 0; i < shape[0]; ++i) {
            for (int64_t j = 0; j < shape[1]; ++j) {
                this->mt[i][j] = qnn::copy(mt[i][j]);
            }
        }
    }

    Qnmat* copy() const {
        return copy(*this);
    }
};

void qnmat_init() {
    global n, N, scly, isys;
    n = crs::n;
    N = crs::N;
    isys = crs::isys;
    if (isys == 3) {
        scly = 2.0 * std::sin(M_PI / 5);
    } else {
        scly = 1.0;
    }
}

Qnmat* zerom(const std::vector<int64_t>& shape) {
    return Qnmat::create(shape);
}

std::vector<Qnmat*> zeroms(const std::vector<int64_t>& shape) {
    int64_t nm = shape[0];
    int64_t n = shape[1];
    std::vector<Qnmat*> qnms(nm);
    for (int64_t i = 0; i < nm; ++i) {
        qnms[i] = zerom({shape[1], shape[2]});
    }
    return qnms;
}

namespace qnn {
    class Qnnum {
        // Implementation of Qnnum class
    public:
        static Qnnum one() {
            // Return an instance representing the number one
        }
        static Qnnum int2qnn(int r) {
            // Convert integer to Qnnum
        }
    };
}

namespace qnv {
    class Qnvec {
        // Implementation of Qnvec class
    public:
        Qnvec(int size) : data(size) {}
        Qnnum& operator[](int index) { return data[index]; }
        const Qnnum& operator[](int index) const { return data[index]; }
        int size() const { return data.size(); }
    private:
        std::vector<qnn::Qnnum> data;
    };

    Qnvec zerov(int size) {
        return Qnvec(size);
    }
}

class Qnmat {
public:
    Qnmat(std::pair<int, int> shape) : rows(shape.first), cols(shape.second), data(shape.first, std::vector<qnn::Qnnum>(shape.second)) {}
    Qnmat(int size) : Qnmat({size, size}) {}

    std::vector<qnn::Qnnum>& operator[](int index) { return data[index]; }
    const std::vector<qnn::Qnnum>& operator[](int index) const { return data[index]; }
    std::pair<int, int> shape() const { return {rows, cols}; }

private:
    int rows, cols;
    std::vector<std::vector<qnn::Qnnum>> data;
};

Qnmat zerom(std::pair<int, int> shape) {
    return Qnmat(shape);
}

Qnmat anym(const std::vector<std::vector<qnn::Qnnum>>& m) {
    auto shape = std::make_pair(m.size(), m[0].size());
    Qnmat m1(shape);
    for (int i = 0; i < shape.first; ++i) {
        for (int j = 0; j < shape.second; ++j) {
            m1[i][j] = m[i][j];
        }
    }
    return m1;
}

Qnmat unitm(int n_) {
    qnn::Qnnum qn1 = qnn::Qnnum::one();
    Qnmat qnm = zerom({n_, n_});
    for (int i = 0; i < n_; ++i) {
        qnm[i][i] = qn1;
    }
    return qnm;
}

Qnmat matrix_2d(const qnv::Qnvec& v1, const qnv::Qnvec& v2) {
    Qnmat m = zerom({2, 2});
    for (int i = 0; i < 2; ++i) {
        m[0][i] = v1[i];
        m[1][i] = v2[i];
    }
    return m;
}

Qnmat matrix_3d(const qnv::Qnvec& v1, const qnv::Qnvec& v2, const qnv::Qnvec& v3) {
    Qnmat m = zerom({3, 3});
    for (int i = 0; i < 3; ++i) {
        m[0][i] = v1[i];
        m[1][i] = v2[i];
        m[2][i] = v3[i];
    }
    return m;
}

Qnmat copy(const Qnmat& m) {
    auto shape = m.shape();
    Qnmat m1(shape);
    for (int i = 0; i < shape.first; ++i) {
        for (int j = 0; j < shape.second; ++j) {
            m1[i][j] = m[i][j];
        }
    }
    return m1;
}

Qnmat copyms(const Qnmat& ms) {
    auto shape = ms.shape();
    std::cout << "ms.shape: " << shape.first << ", " << shape.second << std::endl;  // for test
    Qnmat m1s(shape);
    for (int i = 0; i < shape.first; ++i) {
        m1s[i] = ms[i];
    }
    return m1s;
}

Qnmat int2qnm(const std::vector<std::vector<int>>& r, int n_) {
    Qnmat qnr(n_); // n_ x n_ matrix
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < n_; ++j) {
            qnr[i][j] = qnn::Qnnum::int2qnn(r[i][j]);
        }
    }
    return qnr;
}

qnv::Qnvec qnm2qnv(const Qnmat& qnm) {
    if (qnm.shape().first != 1) {
        throw std::runtime_error("size(shape) != 1 so cannot convert to Qnvec");
    }
    int n = qnm.shape().first;
    qnv::Qnvec qnvt = qnv::zerov(n);
    for (int i = 0; i < n; ++i) {
        qnvt[i] = qnm[i];
    }
    return qnvt;
}

Qnmat add(const Qnmat& ma1, const Qnmat& ma2) {
    auto shape = ma1.shape();
    int n1 = ma1.shape().first;
    int n2 = ma2.shape().first;
    Qnmat a(shape);
    if (n1 == 1 && n2 == 1) { // vectors
        for (int i = 0; i < shape.first; ++i) {
            a[i] = ma1[i] + ma2[i];  // add(v1[i],v2[i])
        }
        return a;
    } else if (n1 == 2 && n2 == 2) { // matrices
        for (int i = 0; i < shape.first; ++i) {
            for (int j = 0; j < shape.second; ++j) {
                a[i][j] = ma1[i][j] + ma2[i][j];  // add(v1[i],v2[i])
            }
        }
        return a;
    }
    throw std::invalid_argument("Incompatible shapes for addition");
}

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



namespace qnn {
    struct Qnnum {
        std::array<double, 3> n;
    };

    Qnnum int2qn(int a, int N) {
        // Implementation of int2qn function
        Qnnum qn;
        // Convert integer to Qnnum logic here
        return qn;
    }

    std::vector<double> qn2npa(const Qnnum& qn) {
        return {qn.n[0], qn.n[1], qn.n[2]};
    }
}

using Qnmat = std::vector<std::vector<qnn::Qnnum>>;

Qnmat qnm2npa(const Qnmat& a) {
    // Qnmatrix to np.array converter
    auto shape = a.size();
    Qnmat b(shape, std::vector<qnn::Qnnum>(a[0].size()));
    for (size_t i = 0; i < shape; ++i) {
        for (size_t j = 0; j < a[i].size(); ++j) {
            b[i][j] = {a[i][j].n[0], a[i][j].n[1], a[i][j].n[2]};
        }
    }
    return b;
}

Qnmat qnm2flt(const Qnmat& a) {
    auto shape = a.size();
    Qnmat b(shape, std::vector<qnn::Qnnum>(a[0].size()));
    for (size_t i = 0; i < shape; ++i) {
        for (size_t j = 0; j < a[i].size(); ++j) {
            b[i][j] = {(a[i][j].n[0] + a[i][j].n[1] * std::sqrt(N)) / a[i][j].n[2]};
        }
    }
    return b;
}

Qnmat intm2qnm(const std::vector<std::vector<int>>& a, int N) {
    auto shape = a.size();
    Qnmat b(shape, std::vector<qnn::Qnnum>(a[0].size()));
    for (size_t i = 0; i < shape; ++i) {
        for (size_t j = 0; j < a[i].size(); ++j) {
            b[i][j] = qnn::int2qn(a[i][j], N);
        }
    }
    return b;
}

void printqnm(const std::string& str, const Qnmat& qnm) {
    size_t ndim = qnm.size() > 0 ? qnm[0].size() > 0 ? 2 : 1 : 0;
    std::cout << str << std::endl;
    if (ndim == 1) {
        for (const auto& row : qnm) {
            std::cout << "[ ";
            for (const auto& elem : row) {
                auto npa = qnn::qn2npa(elem);
                std::cout << "[ " << npa[0] << " " << npa[1] << " " << npa[2] << " ] ";
            }
            std::cout << "]" << std::endl;
        }
    } else if (ndim == 2) {
        for (const auto& row : qnm) {
            std::cout << "[ ";
            for (const auto& elem : row) {
                auto npa = qnn::qn2npa(elem);
                std::cout << "[ " << npa[0] << " " << npa[1] << " " << npa[2] << " ] ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << std::endl;
    } else {
        std::cerr << "ord in printqnm should be 1 or 2 but " << ndim << std::endl;
        exit(1);
    }
}

void printfm(const std::string& str, const std::vector<std::vector<std::vector<double>>>& fm) {
    size_t ndim = fm.size() > 0 ? fm[0].size() > 0 ? 2 : 1 : 0;
    std::cout << str << std::endl;
    if (ndim == 1) {
        for (const auto& row : fm) {
            std::cout << "[ ";
            for (const auto& elem : row) {
                std::cout << elem << " ";
            }
            std::cout << "]" << std::endl;
        }
    } else if (ndim == 2) {
        for (const auto& row : fm) {
            std::cout << "[ ";
            for (const auto& elem : row) {
                std::cout << elem << " ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << std::endl;
    } else {
        std::cerr << "ord in printfm should be 1 or 2 but " << ndim << std::endl;
        exit(1);
    }
}
