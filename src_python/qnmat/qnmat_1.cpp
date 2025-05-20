#include <iostream>
#include <vector>
#include <cmath>
#include "crsys.h"
#include "qnnum.h"
#include "qnvec.h"
#include "qnndarray.h"

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

