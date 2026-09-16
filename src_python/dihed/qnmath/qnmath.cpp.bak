#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>    // For std::abs
#include <cstdlib>  // For std::exit
#include <numeric>  // For std::gcd (C++17)
#include <utility>  // For std::pair
#include <string>
#include <tuple>    // For std::tie

// Forward declarations
//class Qnnum;
//class Qnvec;
//class Qnmat;
#include "qnnum.hpp"
#include "qnvec.hpp" // this should be replaced by #include geometry.hpp
#include "qnmat.hpp" // this can be limitted to matrices to template<double> and template<qnnum>

// Helper for Qnnum (GCD)
//long long gcd(long long a, long long b) {
int64_t gcd(int64_t a, it64_t b) {
    // Handle potential negative inputs for std::gcd
    return std::abs(std::gcd(a, b));
}

// crs struct (global state)
struct crs {
    static int64_t n;
    static int64_t N;
    static int64_t isys;
};

// Define static members
int64_t crs::n;
int64_t crs::N;
int64_t crs::isys;

/**
// Qnnum class (rational number + extra)
class Qnnum {
public:
    int64_t num;
    int64_t extra;
    int64_t den;

    // Constructor from components
    Qnnum(int64_t n = 0, int64_t e = 0, int64_t d = 1) : num(n), extra(e), den(d) {
        if (den == 0) {
            std::cerr << "Error: Qnnum denominator is zero!" << std::endl;
            std::exit(1);
        }
        simplify();
    }

    // Constructor from vector/initializer list
    Qnnum(const std::vector<int64_t>& v) {
        if (v.size() != 3) {
            std::cerr << "Error: Qnnum constructor expects a vector of size 3." << std::endl;
            std::exit(1);
        }
        num = v[0];
        extra = v[1];
        den = v[2];
        if (den == 0) {
            std::cerr << "Error: Qnnum denominator is zero!" << std::endl;
            std::exit(1);
        }
        simplify();
    }

    // Copy method (matches Python's qnn.copy)
    Qnnum copy() const {
        return *this; // Default copy constructor is sufficient
    }

    // Overload operators
    bool operator<(const Qnnum& other) const {
        // Compare a/c < a'/c' => a*c' < a'*c (assuming positive denominators)
        return num * other.den < other.num * den;
    }

    bool operator>=(const Qnnum& other) const {
        return !(*this < other);
    }

    bool operator>(const Qnnum& other) const {
        return num * other.den > other.num * den;
    }

    bool operator<=(const Qnnum& other) const {
        return !(*this > other);
    }

    bool operator==(const Qnnum& other) const {
        // Simplified fractions are equal if num, den, and extra are equal
        return num == other.num && den == other.den && extra == other.extra;
    }

    bool operator!=(const Qnnum& other) const {
        return !(*this == other);
    }

    Qnnum operator-() const { // Unary minus
        return Qnnum(-num, extra, den);
    }

    Qnnum operator+(const Qnnum& other) const {
        // a/c + a'/c' = (a*c' + a'*c) / (c*c')
        int64_t res_num = num * other.den + other.num * den;
        int64_t res_den = den * other.den;
        // Assume 'extra' from left operand
        return Qnnum(res_num, extra, res_den);
    }

    Qnnum operator-(const Qnnum& other) const {
        return *this + (-other);
    }

    Qnnum operator*(const Qnnum& other) const {
        // (a/c) * (a'/c') = (a*a') / (c*c')
        int64_t res_num = num * other.num;
        int64_t res_den = den * other.den;
        // Assume 'extra' from left operand
        return Qnnum(res_num, extra, res_den);
    }

    Qnnum operator/(const Qnnum& other) const {
        // (a/c) / (a'/c') = (a*c') / (c*a')
        if (other.num == 0) {
             std::cerr << "Error: Division by zero Qnnum!" << std::endl;
             std::exit(1);
        }
        int64_t res_num = num * other.den;
        int64_t res_den = den * other.num;
        // Assume 'extra' from left operand
        return Qnnum(res_num, extra, res_den);
    }

    // For printing
    friend std::ostream& operator<<(std::ostream& os, const Qnnum& qn) {
        os << qn.num;
        if (qn.den != 1) {
            os << "/" << qn.den;
        }
        // os << " (e:" << qn.extra << ")"; // Optional: print extra
        return os;
    }

private:
    void simplify() {
        if (den == 0) return; // Should be caught by constructor
        if (num == 0) {
            den = 1;
            return;
        }
        int64_t common = gcd(num, den);
        num /= common;
        den /= common;
        // Ensure denominator is positive
        if (den < 0) {
            num = -num;
            den = -den;
        }
    }
};

// abs function for Qnnum (free function)
Qnnum abs(const Qnnum& a) {
    //N=a.N // Commented out in Python
    Qnnum qn0({0,0,1});
    if (a < qn0) {
        return -a;
    }
    // if (a >= qn0) // This check is redundant, the else path covers it
    return a;
}


// Qnvec class
class Qnvec {
public:
    std::vector<Qnnum> data;
    static std::pair<size_t, size_t> shape; // Static member used in centroid_obj

    // Constructor from size (creates zero vector)
    Qnvec(size_t size = 0) : data(size, Qnnum(0, 0, 1)) {} // Initialize with zero Qnnum

    // Constructor from vector
    Qnvec(const std::vector<Qnnum>& vec) : data(vec) {}

    // Copy method (matches Python's qnv.copy)
    Qnvec copy() const {
        return *this; // Default copy constructor is sufficient
    }

    // Access elements
    Qnnum& operator[](size_t index) {
        return data[index];
    }

    const Qnnum& operator[](size_t index) const {
        return data[index];
    }

    // Get shape (size for 1D vector)
    size_t shape() const {
        return data.size();
    }

    // Get length (same as shape for 1D)
    size_t size() const {
        return data.size();
    }

    // Print method (matches Python's qnv.printqnv)
    void printqnv(const std::string& label) const {
        std::cout << label << ": [";
        for (size_t i = 0; i < data.size(); ++i) {
            std::cout << data[i] << (i == data.size() - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
    }

    // Overload operators
    // Vector-Vector addition
    Qnvec operator+(const Qnvec& other) const {
        if (data.size() != other.data.size()) {
            std::cerr << "Error: Qnvec sizes do not match for addition." << std::endl;
            std::exit(1);
        }
        Qnvec result(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    // Vector-Scalar addition (required by centroid/centroid_obj based on type hints)
    Qnvec operator+(const Qnnum& scalar) const {
        Qnvec result(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] + scalar; // Qnnum + Qnnum
        }
        return result;
    }


    // Vector * Scalar multiplication
    Qnvec operator*(const Qnnum& scalar) const {
        Qnvec result(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }
};

// Initialize static Qnvec::shape (arbitrary default, as its intended value is unknown)
std::pair<size_t, size_t> Qnvec::shape = {0, 0}; // Needs proper initialization based on context not provided


// Qnmat class
class Qnmat {
public:
    std::vector<std::vector<Qnnum>> data;

    // Constructor from shape
    Qnmat(size_t rows = 0, size_t cols = 0) : data(rows, std::vector<Qnnum>(cols, Qnnum(0, 0, 1))) {} // Initialize with zero Qnnum

    // Constructor from shape pair
    Qnmat(std::pair<size_t, size_t> shape) : data(shape.first, std::vector<Qnnum>(shape.second, Qnnum(0, 0, 1))) {}

    // Constructor from vector of vectors
    Qnmat(const std::vector<std::vector<Qnnum>>& mat) : data(mat) {}

    // Copy method (matches Python's qnm.copy)
    Qnmat copy() const {
        return *this; // Default copy constructor is sufficient
    }

    // Access rows
    std::vector<Qnnum>& operator[](size_t row_index) {
        return data[row_index];
    }

    const std::vector<Qnnum>& operator[](size_t row_index) const {
        return data[row_index];
    }

    // Get shape
    std::pair<size_t, size_t> shape() const {
        if (data.empty()) {
            return {0, 0};
        }
        return {data.size(), data[0].size()};
    }

    // Print method (matches Python's qnm.printqnm)
    void printqnm(const std::string& label) const {
        std::cout << label << ":" << std::endl;
        for (size_t i = 0; i < data.size(); ++i) {
            std::cout << "[";
            for (size_t j = 0; j < data[i].size(); ++j) {
                std::cout << data[i][j] << (j == data[i].size() - 1 ? "" : ", ");
            }
            std::cout << "]" << std::endl;
        }
    }
};
**/

// --- Start translating Python functions ---

// Function to initialize global variables
void qnmath_init() {
    crs::n = crs::n; // Literal translation of Python code
    crs::N = crs::N; // Literal translation of Python code
    crs::isys = crs::isys; // Literal translation of Python code
}

// qnmatinv function
Qnmat qnmatinv(const Qnmat& a_i, int64_t n) { // qnmatrix inversion
    // return inversion matrix of a_i
    // n is the order of a_i (nxn qnnumber matrix)
    Qnmat a = a_i.copy();
    //a=np.copy(a_i) // Commented out in Python
    std::vector<Qnnum> pivot(n);
    std::vector<int64_t> ipivot(n);
    std::vector<std::vector<int64_t>> index(n, std::vector<int64_t>(2));
    //N=a[0][0].N // Commented out in Python
    Qnnum qn0({0,0,1});  // 0
    Qnnum qn1({1,0,1});  // 1

    Qnnum det = qn1;  //1.0
    for (int64_t j = 0; j < n; ++j) {
        ipivot[j] = -1;  //ipivot[j]=0
    }

    for (int64_t i = 0; i < n; ++i) {
        Qnnum t = qn0;
        int64_t ir = -1, ic = -1; // Initialize ir, ic

        for (int64_t j = 0; j < n; ++j) {
            if (ipivot[j] == 0) { //if ipivot[j]==1:
                continue;
            }
            for (int64_t k = 0; k < n; ++k) {
                if (ipivot[k] < 0) { //if ipivot[k]-1<0:
                    if (abs(t) >= abs(a[j][k])) {
                        continue;
                    }
                    ir = j;
                    ic = k;
                    t = a[j][k].copy();
                } else if (ipivot[k] > 0) { //elif ipivot[k]-1>0:
                    // Replicate Python's strange return behavior on singularity
                    return a_i.copy(); // Return a copy of the original input matrix
                }
            }
        }

        // Check if pivot element was found. If not, matrix is likely singular.
        if (ir == -1 || ic == -1) {
             std::cerr << "Error: Matrix is singular or algorithm failed to find pivot." << std::endl;
             // Replicate the Python singularity return behavior
             return a_i.copy(); // Return a copy of the original input matrix on error/singularity
        }


        ipivot[ic] = ipivot[ic] + 1;
        if (ir != ic) {
            det = -det;
            for (int64_t l = 0; l < n; ++l) {
                Qnnum swap = a[ir][l].copy();
                a[ir][l] = a[ic][l].copy();
                a[ic][l] = swap;
            }
        }

        index[i][0] = ir;
        index[i][1] = ic;
        pivot[i] = a[ic][ic].copy();

        // Check for zero pivot
        if (pivot[i] == qn0) {
             std::cerr << "Error: Zero pivot encountered. Matrix is singular." << std::endl;
             // Replicate the Python singularity return behavior
             return a_i.copy(); // Return a copy of the original input matrix on error/singularity
        }

        det = det * pivot[i];
        a[ic][ic] = qn1;  //1.0
        for (int64_t l = 0; l < n; ++l) {
            a[ic][l] = a[ic][l] / pivot[i];
        }

        for (int64_t l1 = 0; l1 < n; ++l1) {
            if (l1 == ic) {
                continue;
            }
            Qnnum t_val = a[l1][ic].copy(); // Use a different variable name than the outer loop's 't'
            a[l1][ic] = qn0;  //0.0
            for (int64_t l = 0; l < n; ++l) {
                a[l1][l] = a[l1][l] - a[ic][l] * t_val;
            }
        }
    }

    for (int64_t i = 0; i < n; ++i) {
        int64_t l = n - 1 - i;  // l=n+1-i
        if (index[l][0] == index[l][1]) {
            continue;
        }
        int64_t ir = index[l][0];
        int64_t ic = index[l][1];
        for (int64_t k = 0; k < n; ++k) {
            Qnnum t_val = a[k][ir].copy(); // Use a different variable name
            a[k][ir] = a[k][ic].copy();
            a[k][ic] = t_val;
        }
    }
    //qnm.printqnm("a in qnmatinv",a)  # for test // Commented out in Python
    return a;
}

// fpr check float version
// this should be a function for @ operator
// Note: Python uses np.matrix which is deprecated. Translating to std::vector<std::vector<double>>
std::vector<std::vector<double>> matinv_f(const std::vector<std::vector<double>>& a_i, int64_t n) { // qnmatrix inversion
    // return inversion matrix of a_i
    // n is the order of a (nxn qnnumber matrix)
    std::vector<std::vector<double>> a = a_i; // std::vector copy is deep copy for primitives
    std::vector<double> pivot(n);
    std::vector<int64_t> ipivot(n);
    std::vector<std::vector<int64_t>> index(n, std::vector<int64_t>(2));
    //N=a[0][0].N // Commented out in Python
    double qn0 = 0.0; // for float version
    double qn1 = 1.0;

    double det = qn1;  //1.0
    for (int64_t j = 0; j < n; ++j) {
        ipivot[j] = -1;  //ipivot[j]=0
    }

    for (int64_t i = 0; i < n; ++i) {
        double t = qn0;
        int64_t ir = -1, ic = -1; // Initialize ir, ic

        for (int64_t j = 0; j < n; ++j) {
            if (ipivot[j] == 0) { //if ipivot[j]==1:
                continue;
            }
            for (int64_t k = 0; k < n; ++k) {
                if (ipivot[k] < 0) { //if ipivot[k]-1<0:
                    if (std::abs(t) >= std::abs(a[j][k])) {
                        continue;
                    }
                    ir = j;
                    ic = k;
                    t = a[j][k]; // double assignment is copy
                } else if (ipivot[k] > 0) { //elif ipivot[k]-1>0:
                    // Replicate Python's strange return behavior on singularity
                    return a_i; // Return a copy of the original input matrix
                }
            }
        }

         // Check if pivot element was found (similar logic as qnmatinv)
        if (ir == -1 || ic == -1) {
             std::cerr << "Error: Matrix is singular or algorithm failed to find pivot (float version)." << std::endl;
             // Replicate the Python singularity return behavior
             return a_i; // Return a copy of the original input matrix on error/singularity
        }


        ipivot[ic] = ipivot[ic] + 1;
        if (ir != ic) {
            det = -det;
            for (int64_t l = 0; l < n; ++l) {
                double swap = a[ir][l]; // double assignment is copy
                a[ir][l] = a[ic][l];
                a[ic][l] = swap;
            }
        }

        index[i][0] = ir;
        index[i][1] = ic;
        pivot[i] = a[ic][ic]; // double assignment is copy

        // Check for zero pivot
        if (pivot[i] == qn0) {
             std::cerr << "Error: Zero pivot encountered (float version). Matrix is singular." << std::endl;
             // Replicate the Python singularity return behavior
             return a_i; // Return a copy of the original input matrix on error/singularity
        }

        det = det * pivot[i];
        a[ic][ic] = qn1;  //1.0
        for (int64_t l = 0; l < n; ++l) {
            a[ic][l] = a[ic][l] / pivot[i];
        }

        for (int64_t l1 = 0; l1 < n; ++l1) {
            if (l1 == ic) {
                continue;
            }
            double t_val = a[l1][ic]; // Use a different variable name
            a[l1][ic] = qn0;  //0.0
            for (int64_t l = 0; l < n; ++l) {
                a[l1][l] = a[l1][l] - a[ic][l] * t_val;
            }
        }
    }

    for (int64_t i = 0; i < n; ++i) {
        int64_t l = n - 1 - i;  //l=n+1-i
        if (index[l][0] == index[l][1]) {
            continue;
        }
        int64_t ir = index[l][0];
        int64_t ic = index[l][1];
        for (int64_t k = 0; k < n; ++k) {
            double t_val = a[k][ir]; // Use a different variable name
            a[k][ir] = a[k][ic]; // double assignment is copy
            a[k][ic] = t_val;
        }
    }
    return a;
}

// Helper function for qsort
std::pair<int64_t, int64_t> setlrs(int64_t s, const std::vector<std::vector<int64_t>>& st) {
    //label .l1
    int64_t l = st[s][0];
    int64_t r = st[s][1];
    // s = s - 1; // Python modifies s in the caller via tuple unpacking
    return {l, r}; // Return l and r, caller will update s
}

// Helper function for qsort
struct SetijxtResult {
    int64_t i;
    int64_t j;
    Qnnum xt;
};

SetijxtResult setijxt(int64_t l, int64_t r, const Qnvec& x) {
    //label .l2
    int64_t i = l;
    int64_t j = r;
    int64_t lr = (l + r) / 2; // Integer division
    Qnnum xt = x[lr].copy();
    return {i, j, xt};
}


// qsort function
Qnvec& qsort(Qnvec& x, std::vector<int64_t>& ip, int64_t nx) {
    //     quick sort (ascending order of x)
    //     nx: the number of data x
    //     ip: the initial order
    //     st: a work array

    // xc=np.copy(x) // Python code copies x but then modifies x directly.
    // qnv.printqnv("qnvs",xc) # for test // Commented out in Python

    std::vector<std::vector<int64_t>> st(nx, std::vector<int64_t>(2));
    if (nx == 0) {
        // Python returns None. C++ returns reference to modified x.
        // If nx is 0, x is empty, returning reference to empty x is fine.
        return x;
    }

    //print("nx",nx) # for test // Commented out in Python
    for (int64_t i = 0; i < nx; ++i) {
        ip[i] = i;
    }

    int64_t s = 0;  // s=1
    st[0][0] = 0;     // st[1][0]=1
    st[0][1] = nx - 1;  // st[1][1]=nx

    int64_t l, r;
    // l,r,s=setlrs(s,st) # Python unpacks and modifies s
    std::tie(l, r) = setlrs(s, st);
    s = s - 1; // Manually update s

    int64_t i, j;
    Qnnum xt;
    // i,j,xt=setijxt(l,r,x) # Python unpacks
    SetijxtResult setijxt_res = setijxt(l, r, x);
    i = setijxt_res.i;
    j = setijxt_res.j;
    xt = setijxt_res.xt;

    while (true) {
        //label .l3
        while (true) {
            if (i < nx - 1) { //if i<nx:
                if (x[i] < xt) {
                    i += 1;
                    continue;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        while (true) {
            if (j > 0) { //if j>1:
                if (xt < x[j]) {
                    j -= 1;
                    continue;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if (i <= j) {
            Qnnum temp = x[j].copy();  //temp=xc[j]
            x[j] = x[i].copy();  //xc[j]=qnn.copy(xc[i])
            x[i] = temp.copy();  //xc[i]=temp // Need copy here too
            int64_t itemp = ip[

/*********
this translation is not complete

**/