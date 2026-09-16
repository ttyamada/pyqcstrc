#include <iostream>
#include <vector>
#include "crsys.hpp"
#include "qnnum.hpp"
#include "qnvec.hpp"
#include "qnndarray.hpp"
//#include "qnmat.hpp"

int main() {
    // test
    int isys = 4;
    crsys_init(isys);
    qnnum_init();
    qnvec_init();
    qnndarray_init();
    //qnmat_init();

    int n = crsys_n;
    int N = crsys_N;

    std::vector<int> shape = {n, n};
    //QnNdarray qndm = qnmat::zerom(shape); // nxn qmnum zero matrix
    //std::cout << "qndm.shape: " << qndm.shape() << std::endl;
    //printqndm("zero qnmat", qndm);
    //QnNdarray unitm = qnmat::unitm(n);
    //std::cout << "unitm.shape: " << unitm.shape() << std::endl;
    //printqndm("unit qnmat", unitm);

    //QnNdarray unitmi = copy(unitm); // copy of qnmi
    //std::cout << "unitmi.shape: " << unitmi.shape() << std::endl;
    //printqndm("unitmi", unitmi);

    int nr = 10;
    std::cout << "nr: " << nr << ", n: " << n << std::endl;
    shape = {nr, n, n};
    std::cout << "shape: " << shape[0] << ", " << shape[1] << ", " << shape[2] << std::endl;
    std::vector<QnNdarray> qnda(nr, qnmat::zeros({n, n})); // nr nxn qmnum zero matrices
    std::cout << "qnda.shape: " << qnda[0].shape() << std::endl;

    for (int i = 0; i < nr; i++) {
        std::string str = "qnmat " + std::to_string(i + 1);
        if (i == 0) {
            qnda[i] += unitm;
        } else {
            qnda[i] = qnda[i - 1] + unitm;
        }
        printqndm(str, qnda[i]);
    }

    return 0;
}
