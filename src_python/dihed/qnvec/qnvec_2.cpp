#include <vector>
#include <stdexcept>

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
