//#include <cmath>
//#include <iostream>
#include "crsys.hpp"
#include "number.hpp"
#include "qnnum.hpp"
#include "geometry.hpp"
#include "delaunay.hpp"
#include "wt_off.hpp"

using namespace std;
using namespace delaunay;

double get_det(double v1[],double v2[]) {
    return v1[0]*v2[1]-v1[1]*v2[0];
}

int get_indx(double x, double y, vector<point>& points) {
    double eps=0.000001;
    int n = points.size();
    for (int i=0; i<n; ++i) {
        if (abs(x-points[i].x)<eps && abs(y-points[i].y)<eps) {
            return i;
        }
    }
    cout << " cannt find index of (x, y) in points"<<endl;
    exit(0);
}

int main() // **A minimal example**
{
    // Create 4 input points
    vector<point> points;

double v[][2]={
		{-1,0},
		{0.207107,0},
		{0.207107,0.5},
		{-1.85355,-0.853553},
		{-1.5,-1.20711},
		{-1,1.20711},
		{-1.5,1.20711},
		{-0.146447,-0.853553},
		{0.207107,-0.5},
		{-2.20711,0},
		{-2.20711,-0.5},
		{-0.146447,0.853553},
		{-0.5,1.20711},
		{-1,-1.20711},
		{-0.5,-1.20711},
		{-1.85355,0.853553},
		{-2.20711,0.5}};

    for (int i=0; i<17; ++i) {
        points.emplace_back(v[i][0],v[i][1]);
    }


    int m=points.size();
    cout << "number of points "<<m<<endl;
 
    // Insert them into a triangulation and draw a PDF
    // delaunay.triangulation returns vector<triangle>

    vector<triangle> tria=triangulate(points); // Delaynay trianglation
    int n=tria.size();
    cout <<"number of triangles "<<n<<endl;

    if (n==0) {
        exit(0);
    }

    double v1[2];
    double v2[2];
    double det[n];
    // v1 = b-a v2=c-a

    for (int i=0; i<n; ++i) {
        v1[0]=tria[i].b.x-tria[i].a.x;
        v1[1]=tria[i].b.y-tria[i].a.y;

        v2[0]=tria[i].c.x-tria[i].a.x;
        v2[1]=tria[i].c.y-tria[i].a.y;
        det[i]=get_det(v1, v2);
        cout<<"v1 "<<v1[0]<<" "<<v1[1]<<" "<<v2[0]<<" "<<v2[0]<<" det "<<det[i]<<endl;
    }

    point a,b,c;
    //int i=0;
    double eps=1.e-5;
    double vol=0.0;
    int nzt=0;
    for (int i=0; i<n; ++i) {
        if (abs(det[i])<eps) continue;
        a=tria[i].a; b=tria[i].b; c=tria[i].c; // three points a,b,c
        cout <<a.x<<" "<<a.y<<" "<<b.x<<" "<<b.y<<" "<<c.x<<" "<<c.y<<" det "<<det[i]<<endl;  // for test
        if (abs(det[i])>eps) {
            tria[nzt].a=a; tria[nzt].b=b; tria[nzt].c=c;
            det[nzt]=det[i];
            vol+=abs(det[nzt]);
            ++nzt;
        }
    }
    cout<<"number of finite volume triangles "<<nzt<<" total volume "<<vol<<endl;

    int indx[n][3];
    for (int i=0; i<nzt; ++i) {
        indx[i][0]=get_indx(tria[i].a.x, tria[i].a.y,points);
        indx[i][1]=get_indx(tria[i].b.x, tria[i].b.y,points);
        indx[i][2]=get_indx(tria[i].c.x, tria[i].c.y,points);

        // for .off file
        if (det[i] > 0.0) {
            cout<< 3 << " "<<indx[i][0]<<" "<<indx[i][1]<<" "<<indx[i][2]<<endl;
        } else {
            cout<< 3 <<" "<<indx[i][0]<<" "<<indx[i][2]<<" "<<indx[i][1]<<endl;
        }
    }
    cout<<endl;

    wt_off(v, n, indx, nzt, ".", "dltst");
    return 0;
}

    /**
    double v[][2]={
        {0.9423557,  0.32994074},
        {0.91885435, 0.42031439},
        {0.52627645, 0.18786263},
        {0.36858224, 0.33442723},
        {0.20666006, 0.57436916},
        {0.761603,   0.6413521 },
        {0.12490863, 0.04510419},
        {0.08679162, 0.85320044},
        {0.39410076, 0.31514037},
        {0.94153784, 0.15069643}};
    **/