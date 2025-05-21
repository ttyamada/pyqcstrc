#include <cmath>
#include <iostream>

#include "crsys.hpp"  // for using Qnnumber
#include "number.hpp"
#include "qnnum.hpp"  // for using Qnnumber 
#include "geometry_n.hpp"  // including point<T> for T=<Qnnum>
#include "delaunay_n.hpp"
#include "wt_off.hpp"

using namespace std;
using namespace qnnum;
using namespace delaunay;

// template version for using qnnumber points

// determinant of 2x2 matrix
template<class T>
T get_det(T v1[],T v2[]) {
    return v1[0]*v2[1]-v1[1]*v2[0];
}

// get index of (x,y) in points
template<class T>
int get_indx(T x, T y, vector<point<T>>& points, T eps) {
    int n = points.size();
    for (int i=0; i<n; ++i) {
        //if (abs(x-points[i].x)<eps && abs(y-points[i].y)<eps) {
        if (x==points[i].x && y==points[i].y) {
            return i;
        }
    }
    cout << " cannt find index of (x, y) in points"<<endl;
    exit(0);
}

void test_print(vector<point<Qnnum>>& qnv,int np) {
    for (int i=0; i<np; ++i) {
        //cout << "point "<< qnv[i].x<<" "<< qnv[i].y<< endl;
        printqnn_nlf("point ",qnv[i].x); printqnn(" ",qnv[i].y);
    }
}

bool is_new(point<Qnnum> qnp, std::vector<point<Qnnum>> points, int n) {
    for (int i=0; i<n; ++i) {
        if (qnp == points[i]) {
            return false;
        }
    }
    return true;
}

// independent Qnnum points
std::vector<point<Qnnum>> red_points(Qnnum qnv[][2], int np) {
    std::vector<point<Qnnum>> vec;
    int n = 0;
    for (int i = 0; i < np; ++i) {
        point<Qnnum> pnt(qnv[i][0],qnv[i][1]); // constructor
        if (is_new(pnt, vec, n)) {
            vec.emplace_back(pnt); // error
            n++;  // size of vec
        }
    }
    return vec;
}

// qnnum version (using same template)
int main() {
    int isys=4; // for octagonal
    crsys::crsys_init(isys);
    qnnum::qnnum_init();

    // data for qnnum points (including overlapped points)
    int vt[][2][3]= {
        {{ -1, 0, 1}, {0, 0, 1}},
        {{ -1, 1, 2}, { 0, 0, 1}},
        {{ -1, 1, 2}, { 1, 0, 2}},
        {{ -1, 0, 1}, { 0, 0, 1}},
        {{ -6, -1, 4}, { -2, -1, 4}},
        {{ -3, 0, 2}, { -1, -1, 2}},
        {{ -1, 0, 1}, { 0, 0, 1}},
        {{ -1, 0, 1}, { 1, 1, 2}},
        {{ -3, 0, 2}, { 1, 1, 2}},
        {{ -1, 0, 1}, { 0, 0, 1}},
        {{ -2, 1, 4}, { -2, -1, 4}},
        {{ -1, 1, 2}, { -1, 0, 2}},
        {{ -1, 0, 1}, { 0, 0, 1}},
        {{ -3, -1, 2}, { 0, 0, 1}},
        {{ -3, -1, 2}, { -1, 0, 2}},
        {{ -1, 0, 1}, { 0, 0, 1}},
        {{ -2, 1, 4}, { 2, 1, 4}},
        {{ -1, 0, 2}, { 1, 1, 2}},
        {{ -1, 0, 1}, { 0, 0, 1}},
        {{ -1, 0, 1}, { -1, -1, 2}},
        {{ -1, 0, 2}, { -1, -1, 2}},
        {{ -1, 0, 1}, { 0, 0, 1}},
        {{ -6, -1, 4}, { 2, 1, 4}},
        {{ -3, -1, 2}, { 1, 0, 2}},
        {{ -2, 1, 4}, { 2, 1, 4}},
        {{ -1, 0, 1}, { 0, 0, 1}},
        {{ -1, 1, 2}, { 1, 0, 2}},
        {{ -3, -1, 2}, { 0, 0, 1}},
        {{ -1, 0, 1}, { 0, 0, 1}},
        {{ -3, -1, 2}, { 1, 0, 2}},
        {{ -2, 1, 4}, { -2, -1, 4}},
        {{ -1, 0, 1}, { 0, 0, 1}},
        {{ -1, 0, 2}, { -1, -1, 2}},
        {{ -1, 0, 1}, { 1, 1, 2}},
        {{ -1, 0, 1}, { 0, 0, 1}},
        {{ -1, 0, 2}, { 1, 1, 2}},
        {{ -6, -1, 4}, { -2, -1, 4}},
        {{ -1, 0, 1}, { 0, 0, 1}},
        {{ -3, -1, 2}, { -1, 0, 2}},
        {{ -1, 1, 2}, { 0, 0, 1}},
        {{ -1, 0, 1}, { 0, 0, 1}},
        {{ -1, 1, 2}, { -1, 0, 2}},
        {{ -6, -1, 4}, { 2, 1, 4}},
        {{ -1, 0, 1}, { 0, 0, 1}},
        {{ -3, 0, 2}, { 1, 1, 2}},
        {{ -1, 0, 1}, { -1, -1, 2}},
        {{ -1, 0, 1}, { 0, 0, 1}},
        {{ -3, 0, 2}, { -1, -1, 2}},
        {{ -3, 0, 2}, { 1, 1, 2}},
        {{ -1, 0, 1}, { -1, -1, 2}},
        {{ -1, 0, 1}, { 0, 0, 1}},
        {{ -3, 0, 2}, { -1, -1, 2}}
    };

    //const int n=vt.size();
    const int npt = 52; // 48 or 51 for triangle
    
    Qnnum qnv[npt][2]; // only for qnnumber print
    for (int i=0; i<npt; ++i) {
        for (int j=0; j<2; ++j) {
            qnv[i][j]=Qnnum(vt[i][j]); // qnnumber array
        }
    }

    std::cout<<"npt "<<npt<<std::endl;
    // calculate independent qnnum points
    std::vector<point<Qnnum>> points = red_points(qnv, npt); 
    const int np = points.size();
    std::cout<<"np "<<np<<std::endl;

    test_print(points,np);  // print qnnum coordinates
    double v[np][2];
    vector<point<double>> pointsf;
    for (int i=0; i<np; ++i) {
        double x = qn2flt(points[i].x);
        double y = qn2flt(points[i].y);
        v[i][0]=x; v[i][1]=y;
        //std::cout<<"x "<<points[i].x<< " y "<<points[i].y<<std::endl; // for test
        std::cout<<x<< " "<<y<<std::endl; // for test
        pointsf.emplace_back(x,y);
    }

    // Insert them into a triangulation and draw a PDF
    // delaunay::triangulate returns vector<triangle<Qnnum>>
   
    vector<triangle<Qnnum>> tria=triangulate<Qnnum>(points); // Delaunay triangulation

    const int n=tria.size();
    cout<<"number of triangles "<<n<<'\n';

    if (n==0) {
        exit(0);
    }

    Qnnum v1[2];
    Qnnum v2[2];
    Qnnum det[n];  // determinant
    // v1 = b-a v2=c-a
    for (int i=0; i<n; ++i) {
        v1[0]=tria[i].b.x-tria[i].a.x;
        v1[1]=tria[i].b.y-tria[i].a.y;

        v2[0]=tria[i].c.x-tria[i].a.x;
        v2[1]=tria[i].c.y-tria[i].a.y;
        det[i]=get_det<Qnnum>(v1, v2);
    }

    point<Qnnum> a,b,c;
    Qnnum eps=zero();
    Qnnum vol=zero();
    int nzt=0;
    int i=0;
    for (triangle<Qnnum> t : tria) {
        a=t.a; b=t.b; c=t.c; // three points a,b,c
        cout << qn2flt(a.x)<<" "<<qn2flt(a.y)<<" "<<qn2flt(b.x)<<" "<<qn2flt(b.y)<<" "<<qn2flt(c.x)<<" "<<qn2flt(c.y)<<" ";
        cout <<" det "<<qn2flt(det[i])<<endl;  // for test
        //cout <<" det "<<qn2flt(det[i])<<" "<<a.x<<" "<<a.y<<" "<<b.x<<" "<<b.y<<" "<<c.x<<" "<<c.y<<endl;  // for test
 
        if (abs(det[i])>eps) {
            tria[nzt].a=a; tria[nzt].b=b; tria[nzt].c=c;
            det[nzt]=det[i];
            vol+=abs(det[nzt]);
            ++nzt;
        }
        ++i;
    }
    cout<<"number of finite volume triangles "<<nzt<<" total volume "<<qn2flt(vol)<<"="<<vol<<endl;


    int indx[n][3];
    //Qnnum eps = zero();
    for (int i=0; i<n; ++i) {
        indx[i][0]=get_indx<Qnnum>(tria[i].a.x, tria[i].a.y,points,eps);
        indx[i][1]=get_indx<Qnnum>(tria[i].b.x, tria[i].b.y,points,eps);
        indx[i][2]=get_indx<Qnnum>(tria[i].c.x, tria[i].c.y,points,eps);

        // for .off file
        if (det[i] > zero()) {
            cout<< 3 << " "<<indx[i][0]<<" "<<indx[i][1]<<" "<<indx[i][2]<<endl;
        } else {
            cout<< 3 <<" "<<indx[i][0]<<" "<<indx[i][2]<<" "<<indx[i][1]<<endl;
        }
    }
    cout<<endl;

    wt_off(v, np, indx, n, ".", "dltst_qnn");
    return 0;
}

/**
void test_print(Qnnum qnv[][2],int np) {
    for (int i=0; i<np; ++i) {
        cout << "point "<< qn2flt(qnv[i][0])<<" "<< qn2flt(qnv[i][1])<< endl;
    }
}

bool is_new(Qnnum qnv[2],std::vector<Qnnum[2]> vec, int n) {
    for (int i=0; i<n; ++i) {
        if (qnv[0] != vec[i][0] || qnv[1] != vec[i][1]) {
            return false;
        }
    }
    return true;
}

std::vector<Qnnum[2]> red_points(qnv[][2], int np) {
    std::vector<Qnnum[2]> vec;
    int n = 0;
    for (int i = 0; i < np; ++i) {
        if (is_new(qnv[i], vec, n) {
            vec.push_back(i);
            n++;  // size of vec
    }
    return vec;
}

**/
