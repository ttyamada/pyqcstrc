#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <numeric>
#include "qnnum.hpp"

using namespace qnnum;
using namespace std;

// Main function for testing or demonstration
int main() {
    init();
    Qnnum a = one();
    Qnnum b = zero();
    int d_[] = {2,0,1};
    Qnnum d = any(d_);
    Qnnum c = a + d;
    Qnnum e = d - a;
    
    cout << "a+d "<< "[" << c.n[0] << ", " << c.n[1] << ", " << c.n[2] << "]"<< endl;
    cout << "d-a "<< "[" << e.n[0] << ", " << e.n[1] << ", " << e.n[2] << "]"<< endl;
    return 0;
}
