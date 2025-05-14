#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>

using namespace std;

int generator_off_dim4_triangle(const vector<vector<double>>& pnt, 
    const vector<vector<int>> trg, const string& path, const string& filename) {
    ofstream f(path + "/" + filename + ".off");
    f << "OFF"<<endl;
    f << obj.size() << " "<< ntr << " 0" <<endl;  // number of independent points
    //f << filename << endl;

    int n = crs::n;
    int isys = crs::isys;
    int ni = (isys > 2) ? 2 : 3;  // 2 or 3 for dihedral or icosahedral 
    for (int i = 0; i < trg.size(); ++i) {  // i1-th triangle
        f << obj[i][0] << " " << obj[i][1] << " 0.0"<<endl;
    }
    for (int j = 0; i < ; ++j) {  // j-th triangle
        f << "  " << trg[j][0] << " " << trg[i][1] << " " << trg[j][2];
    }
    f.close();
    return 0;
}

