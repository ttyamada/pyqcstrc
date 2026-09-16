#include <vector>
#include <array>
#include "crsys.hpp"
#include "qnnum.hpp"
#include "qnvec.hpp"
#include "qnndarray.hpp"

namespace crsys {
    extern int isys;
    extern int n;
    extern int N;
}

namespace qnn {
    // Assuming zero() returns a zero-initialized object of some type
    int zero() {
        return 0; // Placeholder for actual implementation
    }
}

namespace qnv {
    // Placeholder for Qnvec class
    class Qnvec {};
}

namespace qna {
    class QnNdarray {
    public:
        virtual ~QnNdarray() = default;
    };

    std::vector<QnNdarray> zeros(const std::array<int, 2>& shape, const QnNdarray& dtype) {
        return std::vector<QnNdarray>(shape[0], dtype);
    }
}

class Qntri : public qna::QnNdarray {
public:
    static int n_i;
    static int N;
    static std::array<int, 2> shape;

    Qntri() {
        int qn0 = qnn::zero();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < n_i; ++j) {
                data[i][j] = qn0;
            }
        }
    }

    static Qntri create() {
        shape = {3, n_i};
        return Qntri();
    }

    bool operator==(const Qntri& other) const {
        return eq(*this, other);
    }

    bool operator!=(const Qntri& other) const {
        return not_eq(*this, other);
    }

    std::array<int, 2> get_shape() const {
        return shape;
    }

private:
    std::array<std::array<int, 10>, 3> data; // Assuming n_i is at most 10 for simplicity

    static bool eq(const Qntri& qnv1, const Qntri& qnv2) {
        for (int i = 0; i < 3; ++i) {
            if (qnv1.data[i] != qnv2.data[i]) {
                return false;
            }
        }
        return true;
    }

    static bool not_eq(const Qntri& qnv1, const Qntri& qnv2) {
        for (int i = 0; i < 3; ++i) {
            if (qnv1.data[i] != qnv2.data[i]) {
                return true;
            }
        }
        return false;
    }
};

int Qntri::n_i;
int Qntri::N;
std::array<int, 2> Qntri::shape;

void qntri_init() {
    using namespace crsys;
    if (isys == 2) {
        Qntri::n_i = 3; 
    } else {
        Qntri::n_i = 2; 
    }
}

Qntri zerotri(int n_i) {
    int qn0 = qnn::zero();
    Qntri v1 = Qntri::create();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < n_i; ++j) {
            v1.data[i][j] = qn0;
        }
    }
    return v1;
}

std::vector<Qntri> zerotris(const std::array<int, 2>& shape) {
    std::cout << "shape in zerovs: " << shape[0] << ", " << shape[1] << std::endl; // for test
    int nv = shape[0];
    int n = shape[1];
    std::vector<Qntri> qntrs = qna::zeros({nv, 1}, Qntri::create());
    for (int i = 0; i < nv; ++i) {
        qntrs[i] = zerotri(n);
    }
    return qntrs;
}

Qntri anytri(const std::vector<qnv::Qnvec>& v) {
    int n = v.size();
    std::cout << "n: " << n << std::endl; // for test
    Qntri v1 = Qntri::create();
    std::cout << "v1.shape: " << v1.get_shape()[0] << ", " << v1.get_shape()[1] << std::endl; // for test
    for (int i = 0; i < 3; ++i) {
        v1.data[i] = v[i]; // Assuming Qnvec can be assigned directly
    }
    return v1;
}

void wt_qntri(const Qntri& tri) {
    auto shap = tri.get_shape();
    std::cout << "shape: " << shap[0] << ", " << shap[1] << std::endl;
    for (int i = 0; i < 3; ++i) {
        // Assuming printqnv is a function that prints Qnvec
        // qnv::printqnv("tri", tri.data[i]);
    }
}

