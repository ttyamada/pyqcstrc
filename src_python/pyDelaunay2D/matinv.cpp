#include <iostream>
#include <vector>
#include <cmath>

std::allocator<int> alloc;

void matinv0(const std::vector<Qnnum>& a, int n1, int n2, Qnnum& b, int m, float& det, int& ipivot, int& index, Qnnum& pivot);

void matinvnd(const std::vector<Qnnum>& a, Qnnum& b, int m, float& det) {
    shape = a.shape();
    n1=shape[0];
    n2=shape[1];
    std::vector<int> ipivot(n1);
    std::vector<int> index(n1,2);
    std::vector<Qnnum> pivot(n1);
    
    matinv0(a, n1, n2, b, m, det, ipivot, index, pivot);
}

void matinv0(std::vector<Qnnum>& a, int nm, int n, Qnnum& b, int m, float& determ, int& ipivot, int& index, Qnnum& pivot) {
    float amax, swap, t;
    determ = 1.0;
    
    for (int j = 0; j < n; j++) {
        ipivot[j] = 0;
    }
    
    for (int i = 0; i < n; i++) {
        amax = 0.0;
        int irow = 0, icolum = 0;
        
        for (int j = 0; j < n; j++) {
            if (ipivot[j] == 1) continue;
            for (int k = 0; k < n; k++) {
                if (ipivot[k] - 1 < 0) {
                    if (std::abs(amax) >= std::abs(a[j][k])) continue;
                    irow = j;
                    icolum = k;
                    amax = a[j][k];
                } else if (ipivot[k] - 1 > 0) {
                    return;
                }
            }
        }
        
        ipivot[icolum]++;
        if (irow != icolum) {
            determ = -determ;
            for (int l = 0; l < n; l++) {
                swap = a[irow][l];
                a[irow][l] = a[icolum][l];
                a[icolum][l] = swap;
            }
            if (m != 0) {
                swap = b[irow];
                b[irow] = b[icolum];
                b[icolum] = swap;
            }
        }
        
        index[i][0] = irow;
        index[i][1] = icolum;
        pivot[i] = a[icolum][icolum];
        determ *= pivot[i];
        a[icolum][icolum] = 1.0;
        
        for (int l = 0; l < n; l++) {
            a[icolum][l] /= pivot[i];
        }
        
        if (m != 0) {
            b[icolum] /= pivot[i];
        }
        
        for (int l1 = 0; l1 < n; l1++) {
            if (l1 == icolum) continue;
            amax = a[l1][icolum];
            a[l1][icolum] = 0.0;
            for (int l = 0; l < n; l++) {
                a[l1][l] -= a[icolum][l] * amax;
            }
            if (m != 0) {
                b[l1] -= b[icolum] * amax;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        int l = n - 1 - i;
        if (index[l][0] == index[l][1]) continue;
        int jrow = index[l][0];
        int jcolum = index[l][1];
        for (int k = 0; k < n; k++) {
            swap = a[k][jrow];
            a[k][jrow] = a[k][jcolum];
            a[k][jcolum] = swap;
        }
    }
}
