#include <iostream>
#include <vector>
#include <array>
#include <cstring>

class QnNdarray {
public:
    QnNdarray(const std::array<int, 3>& shape) : shape(shape) {
        data.resize(shape[0] * shape[1] * shape[2]);
    }

    int& operator()(int i, int j, int k) {
        return data[i * shape[1] * shape[2] + j * shape[2] + k];
    }

private:
    std::array<int, 3> shape;
    std::vector<int> data;
};

class Qnsym_Octa : public QnNdarray {
public:
    Qnsym_Octa() : QnNdarray({nr, n, n}) {
        int ng = 3; // three generating elements
        std::array<std::array<std::array<int, 5>, 5>, 3> rg = {};
        std::array<int, 2> gord = {8, 2}; // gord=(8,2,2)

        rg[0][0][1] = 1; rg[0][1][2] = 1; rg[0][2][3] = 1; rg[0][3][0] = -1; rg[0][4][4] = 1; // R8 
        rg[1][1][2] = 1; rg[1][2][1] = 1; rg[1][0][3] = 1; rg[1][3][0] = 1; rg[1][4][4] = 1; // M

        std::vector<std::vector<std::vector<int>>> r(nr, std::vector<std::vector<int>>(n, std::vector<int>(n, 0))); // nD int array
        std::vector<std::vector<std::vector<int>>> rt(nr, std::vector<std::vector<int>>(n, std::vector<int>(n, 0))); // nD int array
        set_r(rg, gord, r, rt); // set all integer symmetry operators r

        this->r = r;  // integer rotation matrix
        this->rt = rt;
        this->r_qn0 = get_r_qn0(r, nr);  // symmetry operator for qnv r_qn0@qnv
        this->r_qn = rtor_qn(r);
        this->r_qn_e = rtor_qn_e(r);
        this->r_qn_i = rtor_qn_i(r);
        this->nr = nr;
        this->n = n;
        this->N = N;
        this->shape = {nr, n, n};
        std::vector<std::vector<int>> mpltbl(nr, std::vector<int>(nr, 0));
        set_mpltbl(mpltbl, r);
        this->mpltbl = mpltbl;
    }

private:
    std::vector<std::vector<std::vector<int>>> r;
    std::vector<std::vector<std::vector<int>>> rt;
    std::vector<int> r_qn0;
    std::vector<int> r_qn;
    std::vector<int> r_qn_e;
    std::vector<int> r_qn_i;
    int nr;
    int n;
    int N;
    std::array<int, 3> shape;
    std::vector<std::vector<int>> mpltbl;
};

class Qnsym_Deca : public QnNdarray {
public:
    Qnsym_Deca() : QnNdarray({nr, n, n}) {
        int ng = 3; // three generating elements
        std::array<std::array<std::array<int, 5>, 5>, 3> rg = {};
        std::array<int, 3> gord = {5, 2, 2}; // gord=(10,2,2)

        rg[0][0][1] = 1; rg[0][1][2] = 1; rg[0][2][3] = 1; // R5
        rg[0][3][0] = -1; rg[0][3][1] = -1; rg[0][3][2] = -1; rg[0][3][3] = -1;
        rg[0][4][4] = 1;
        rg[1][0][3] = 1; rg[1][3][0] = 1; rg[1][1][2] = 1; rg[1][2][1] = 1; rg[1][4][4] = 1; // M
        rg[2][0][0] = -1; rg[2][1][1] = -1; rg[2][2][2] = -1; rg[2][3][3] = -1; rg[2][4][4] = -1; // I

        std::vector<std::vector<std::vector<int>>> r(nr, std::vector<std::vector<int>>(n, std::vector<int>(n, 0)));
        std::vector<std::vector<std::vector<int>>> rt(nr, std::vector<std::vector<int>>(n, std::vector<int>(n, 0)));
        set_r(rg, gord, r, rt); // set all integer rotation matrices

        this->r = r;
        this->rt = rt;
        this->r_qn0 = get_r_qn0(r, nr);
        this->r_qn = rtor_qn(r);
        this->r_qn_e = rtor_qn_e(r);
        this->r_qn_i = rtor_qn_i(r);
        this->nr = nr;
        this->n = n;
        this->N = N;
        this->shape = {nr, n, n};
        std::vector<std::vector<int>> mpltbl(nr, std::vector<int>(nr, 0));
        set_mpltbl(mpltbl, r);
        this->mpltbl = mpltbl;
    }

private:
    std::vector<std::vector<std::vector<int>>> r;
    std::vector<std::vector<std::vector<int>>> rt;
    std::vector<int> r_qn0;
    std::vector<int> r_qn;
    std::vector<int> r_qn_e;
    std::vector<int> r_qn_i;
    int nr;
    int n;
    int N;
    std::array<int, 3> shape;
    std::vector<std::vector<int>> mpltbl;
};

