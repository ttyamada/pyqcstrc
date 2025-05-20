#include <iostream>
#include <vector>
#include <stdexcept>

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

