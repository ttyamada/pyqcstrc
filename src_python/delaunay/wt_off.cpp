#include "wt_off.hpp"

using namespace std;

void wt_off(double pnt[][2], int np, int trg[][3], int nt, const string& path, const string& filename) {
    ofstream f(path + "/" + filename + ".off");
    f << "OFF"<<endl;
    f << np << " "<< nt << " 0" <<endl;  // number of independent points
    //f << filename << endl;
    f << endl;
    for (int i = 0; i < np; ++i) {  // i1-th point
        f << pnt[i][0] << " " << pnt[i][1] << " 0.0"<<endl;
    }
    f << endl;
    for (int j = 0; j <nt ; ++j) {  // j-th triangle
        f << "3 " << trg[j][0] << " " << trg[j][1] << " " << trg[j][2] << endl;
    }
    f.close();
    return;
}

