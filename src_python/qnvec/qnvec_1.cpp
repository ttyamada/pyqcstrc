#include <iostream>
#include <vector>
#include <type_traits>
#include "crsys.hpp"
#include "qnnum.hpp"
#include "qnndarray.hpp"

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
