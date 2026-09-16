#include <iostream>
#include <vector>
#include <array>
#include <algorithm>

namespace crs {
    // Placeholder for crsys functionality
}

namespace qnn {
    class Qnnum {
    public:
        static Qnnum zero() {
            return Qnnum(0);
        }
        Qnnum(int value) : value(value) {}
        bool operator>=(const Qnnum& other) const {
            return value >= other.value;
        }
    private:
        int value;
    };
}

namespace qnv {
    using Qnvec = std::array<double, 2>;

    Qnvec copy(const Qnvec& vec) {
        return vec;
    }

    Qnvec zerov(size_t size) {
        return Qnvec{0.0, 0.0}; // Adjust size as needed
    }

    std::vector<Qnvec> zerovs(const std::array<size_t, 2>& shape) {
        return std::vector<Qnvec>(shape[0], Qnvec{0.0, 0.0});
    }
}

namespace qna {
    using QnNdarray = std::vector<std::array<std::array<double, 2>, 3>>;

    QnNdarray zeros(const std::array<size_t, 3>& shape) {
        return QnNdarray(shape[0], std::array<std::array<double, 2>, 3>{{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}});
    }
}

std::array<std::array<std::array<double, 2>, 2>, 3> get_edge(const qna::QnNdarray& tri) {
    auto edge = qna::zeros({3, 2, 2});
    for (size_t i = 0; i < 3; ++i) {
        size_t j = (i + 1) % 3;
        for (size_t k = 0; k < 2; ++k) {
            edge[i][0][k] = tri[i][k];
            edge[i][1][k] = tri[j][k];
        }
    }
    return edge;
}

qnn::Qnnum det_vecabc(const qnv::Qnvec& a, const qnv::Qnvec& b, const qnv::Qnvec& c) {
    qnv::Qnvec da = {a[0] - c[0], a[1] - c[1]};
    qnv::Qnvec db = {b[0] - c[0], b[1] - c[1]};
    return qnn::Qnnum(da[0] * db[1] - db[0] * da[1]);
}

qna::QnNdarray counter_clockwise(qna::QnNdarray tri) {
    if (det_vecabc(tri[0], tri[1], tri[2]) > qnn::Qnnum::zero()) {
        return tri;
    } else {
        std::swap(tri[0], tri[1]);
        return tri;
    }
}

std::pair<int, std::vector<qnv::Qnvec>> common_points(const qna::QnNdarray& tri_1, const qna::QnNdarray& tri_2) {
    qnv::Qnvec det1 = qnv::zerov(3);
    qnv::Qnvec det2 = qnv::zerov(3);
    auto comx = qnv::zerovs({6, 2});
    int n = 0;

    for (size_t k = 0; k < 3; ++k) {
        for (size_t i = 0; i < 3; ++i) {
            size_t j = (i + 1) % 3;
            det1[i] = det_vecabc(tri_1[i], tri_1[j], tri_2[k]);
        }
        if (det1[0] >= qnn::Qnnum::zero() && det1[1] >= qnn::Qnnum::zero() && det1[2] >= qnn::Qnnum::zero()) {
            comx[n] = tri_2[k];
            n++;
        }
        for (size_t i = 0; i < 3; ++i) {
            size_t j = (i + 1) % 3;
            det2[i] = det_vecabc(tri_2[i], tri_2[j], tri_1[k]);
        }
        if (det2[0] >= qnn::Qnnum::zero() && det2[1] >= qnn::Qnnum::zero() && det2[2] >= qnn::Qnnum::zero()) {
            comx[n] = tri_1[k];
            n++;
        }
    }
    return {n, std::vector<qnv::Qnvec>(comx.begin(), comx.begin() + n)};
}

std::pair<int, std::vector<qnv::Qnvec>> rmv_overlapedx(std::vector<qnv::Qnvec>& x, int n0) {
    int n = 0;
    for (int i = 0; i < n0; ++i) {
        if (i == 0) {
            n++;
            continue;
        }
        bool iskp = false;
        for (int j = 0; j < n; ++j) {
            if (x[i][0] == x[j][0] && x[i][1] == x[j][1]) {
                iskp = true;
                break;
            }
        }
        if (!iskp) {
            x[n] = x[i];
            n++;
        }
    }
    return {n, std::vector<qnv::Qnvec>(x.begin(), x.begin() + n)};
}
