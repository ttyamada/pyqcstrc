#include <iostream>
#include <vector>
#include "crsys.hpp"
#include "qnnum.hpp"
//#include "qnvec.hpp"
//#include "qnndarray.hpp"
#include "qnmat.hpp"

// this hould be independent of qnvec and qnndarray
// use geometry instead (point edge triangle tetrahedron)

void qnmat_tst(const std::string& str, int64_t isys) {
    std::cout << str << std::endl;
    crsys_init(isys);
    qnnum_init();
    qnvec_init();
    qnmat_init();
    
    int n = crsys::n;
    int N = crsys::N;
    std::cout << "n " << n << " N " << N << std::endl;
    std::vector<std::vector<Qnnum>> qnm = zerom(n, n); // nxn qmnum zero matrix
    std::cout << "qnm.ndim " << 2 << std::endl; // Assuming 2D
    std::cout << "qnm.shape " << n << " " << n << std::endl;
    printqnm("qnm", qnm);

    auto qnm1 = copy(qnm);
    std::cout << "qnm1.ndim " << 2 << std::endl; // Assuming 2D
    std::cout << "qnm1.shape " << n << " " << n << std::endl;
    printqnm("qnm1", qnm1);

    auto unm1 = unitm(n); // nxn qmnum unit matrix
    std::cout << "unm1.ndim " << 2 << std::endl; // Assuming 2D
    std::cout << "unm1.shape " << n << " " << n << std::endl;
    printqnm("unm1", unm1);

    auto unm2 = unitm(n); // nxn qmnum unit matrix
    std::cout << "unm2.ndim " << 2 << std::endl; // Assuming 2D
    std::cout << "unm2.shape " << n << " " << n << std::endl;
    printqnm("unm2", unm2);

    auto unm3 = matrix_multiply(unm2, unm1);
    printqnm("unm2@unm1", unm3);
    
    Qnnum M0({0, 0, 1});
    Qnnum M1({1, 0, 1});
    Qnnum M2({0, 1, 1});
    Qnnum M3({1, 1, 2});
    Qnnum M4({1, -1, 2});
    std::vector<Qnnum> qnv1;
    if (n == 5) {
        qnv1 = anyv({M0, M1, M2, M3, M4});
    } else if (n == 6) {
        qnv1 = anyv({M0, M1, M2, M3, M4, M0});
    }
    printqnv("qnv1", qnv1);
    auto qnv2 = matrix_multiply(unm2, qnv1);
    printqnv("qnv2", qnv2);
}

int main() {
    int64_t isys = 3;
    qnmat_tst("decagonal", isys);

    isys = 2;
    qnmat_tst("icosahedral", isys);

    return 0;
}

