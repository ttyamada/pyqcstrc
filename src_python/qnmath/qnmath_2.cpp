#include <vector>
#include <cmath>
#include <numeric> // Not strictly needed for this code, but often useful with vectors

// fpr check float version     
// this should be a function for @ operator  
// qnmatrix inversion
// return inversion matrix of a_i
// n is the order of a (nxn qnnumber matrix)
std::vector<std::vector<double>> matinv_f(const std::vector<std::vector<double>>& a_i, long long n) {
    // return inversion matrix of a_i
    // n is the order of a (nxn qnnumber matrix)
    std::vector<std::vector<double>> a = a_i;
    std::vector<double> pivot(n);
    std::vector<long long> ipivot(n); 
    std::vector<std::vector<long long>> index(n, std::vector<long long>(2));
    //N=a[0][0].N
    double qn0 = 0.0; // for float version
    double qn1 = 1.0;
    
    double det = qn1;  //1.0 
    for (long long j = 0; j < n; ++j) {
        ipivot[j] = -1;  //ipivot[j]=0
    }
    
    long long ir = 0; // Declare ir and ic outside the loop
    long long ic = 0; // Declare ir and ic outside the loop

    for (long long i = 0; i < n; ++i) { 
        double t = qn0;
        for (long long j = 0; j < n; ++j) {
            if (ipivot[j] == 0) { //if ipivot[j]==1:
                continue;
            }
            for (long long k = 0; k < n; ++k) {
                if (ipivot[k] < 0) { //if ipivot[k]-1<0:
                    if (std::abs(t) >= std::abs(a[j][k])) {
                        continue;
                    }
                    ir = j;
                    ic = k;
                    t = a[j][k];
                } else if (ipivot[k] > 0) { //elif ipivot[k]-1>0:
                    return a;
                }
            }
        }
    
        ipivot[ic] = ipivot[ic] + 1;
        if (ir != ic) {
            det = -det;
            for (long long l = 0; l < n; ++l) {
                double swap = a[ir][l];
                a[ir][l] = a[ic][l];
                a[ic][l] = swap;
            }
        }

        index[i][0] = ir;
        index[i][1] = ic;
        pivot[i] = a[ic][ic];
        det = det * pivot[i];
        a[ic][ic] = qn1;  //1.0
        for (long long l = 0; l < n; ++l) {
            a[ic][l] = a[ic][l] / pivot[i];
        }

        for (long long l1 = 0; l1 < n; ++l1) {
            if (l1 == ic) {
                continue;
            }
            t = a[l1][ic];
            a[l1][ic] = qn0;  //0.0
            for (long long l = 0; l < n; ++l) {
                a[l1][l] = a[l1][l] - a[ic][l] * t;
            }
        }
    }
    for (long long i = 0; i < n; ++i) {
        long long l = n - 1 - i;  //l=n+1-i
        if (index[l][0] == index[l][1]) {
            continue;
        }
        ir = index[l][0];
        ic = index[l][1];
        for (long long k = 0; k < n; ++k) {
            double t_swap = a[k][ir]; // Use a different variable name to avoid conflict with the 't' used earlier
            a[k][ir] = a[k][ic];
            a[k][ic] = t_swap;
        }
    }
    return a;
}
