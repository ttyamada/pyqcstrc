#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include "qnn.h" // Assuming qnn.h contains the necessary declarations for Qnnum and related functions
#include "crsys.h" // Assuming crsys.h contains the necessary declarations for isys, n, N

class QnNdarray : public std::vector<std::vector<std::vector<Qnnum>>> {
public:
    QnNdarray(std::vector<int> shape) : std::vector<std::vector<std::vector<Qnnum>>>(shape[0], std::vector<std::vector<Qnnum>>(shape[1], std::vector<Qnnum>(shape[2]))) {
        this->shape = shape;
    }

    std::vector<int> shape;
};

void qnndarray_init() {
    global_n = crsys::n;
    global_N = crsys::N;
    isys = crsys::isys;
}

QnNdarray copy(const QnNdarray& qna1) {
    return qna1; // Assuming QnNdarray has a copy constructor
}

QnNdarray zeros(std::vector<int> shape) {
    QnNdarray qna(shape);
    Qnnum qn0 = qnn::zero();
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                qna[i][j][k] = qn0;
            }
        }
    }
    return qna;
}

QnNdarray anya(const std::vector<std::vector<std::vector<Qnnum>>>& vec, std::vector<int> shape) {
    QnNdarray qnva(shape);
    int ndim = shape.size();
    if (ndim == 1) {
        for (int i = 0; i < shape[0]; ++i) {
            qnva[i][0][0] = vec[i][0][0]; // Assuming vec is 1D
        }
    } else if (ndim == 2) {
        for (int i = 0; i < shape[0]; ++i) {
            for (int j = 0; j < shape[1]; ++j) {
                qnva[i][j][0] = vec[i][j][0]; // Assuming vec is 2D
            }
        }
    } else if (ndim == 3) {
        for (int i = 0; i < shape[0]; ++i) {
            for (int j = 0; j < shape[1]; ++j) {
                for (int k = 0; k < shape[2]; ++k) {
                    qnva[i][j][k] = vec[i][j][k]; // Assuming vec is 3D
                }
            }
        }
    }
    return qnva;
}

void printqndm(const std::string& str, const QnNdarray& qnm) {
    int ndim = qnm.shape.size();
    std::cout << str << std::endl;
    if (ndim == 1) {
        std::cout << "[ ";
        for (const auto& item : qnm) {
            std::cout << qnn::qn2npa(item) << " ";
        }
        std::cout << "]" << std::endl;
    } else if (ndim == 2) {
        for (const auto& row : qnm) {
            std::cout << "[ ";
            for (const auto& item : row) {
                std::cout << qnn::qn2npa(item) << " ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << std::endl;
    } else if (ndim == 3) {
        for (const auto& matrix : qnm) {
            std::cout << std::endl;
            for (const auto& row : matrix) {
                std::cout << "[ ";
                for (const auto& item : row) {
                    std::cout << qnn::qn2npa(item) << " ";
                }
                std::cout << "]" << std::endl;
            }
        }
        std::cout << std::endl;
    } else {
        std::cerr << "ord in printqnm should be 1, 2 or 3 but " << ndim << std::endl;
        exit(EXIT_FAILURE);
    }
}

std::vector<Qnnum> insert(std::vector<Qnnum>& index, const Qnnum& point) {
    index.push_back(point);
    return index;
}

std::vector<Qnnum> append(std::vector<Qnnum>& point) {
    point.push_back(Qnnum()); // Assuming default constructor for Qnnum
    return point;
}
