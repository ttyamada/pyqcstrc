#include "qnnum.hpp"

using namespace std;
using namespace qnnum;

// Main function for testing or demonstration
int main() {
    qnnum_init();
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
