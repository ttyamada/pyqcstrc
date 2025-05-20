#include <iostream>
#include <vector>
#include <cmath>
#include <array>

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

