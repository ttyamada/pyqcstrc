#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <string>

// Dependencies and imports
using namespace std;

// Namespace for Qnnum
namespace qnn {
    typedef double Qnnum;
    inline Qnnum zero() { return 0.0; }
    inline Qnnum one() { return 1.0; }
    inline Qnnum inf() { return std::numeric_limits<double>::infinity(); }
    inline void printqnn(const string &s, Qnnum val) {
        cout << s << ": " << val << endl;
    }
}

// Namespace for Qnvec
namespace qnv {
    typedef vector<double> Qnvec;
    inline Qnvec copy(const Qnvec &v) {
        return v;
    }
    inline Qnvec zerov(int n) {
        return Qnvec(n, 0.0);
    }
    inline vector<Qnvec> zerovs(int rows, int cols) {
        return vector<Qnvec>(rows, zerov(cols));
    }
    inline double dot(const Qnvec &a, const Qnvec &b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size() && i < b.size(); i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    inline void printqnv(const string &s, const Qnvec &v) {
        cout << s << ": ";
        for (auto d : v) cout << d << " ";
        cout << endl;
    }
}

// Namespace for QnNdarray.
// For 2D arrays (e.g., a triangle: 3 vertices each a Qnvec)
namespace qna {
    typedef vector<qnv::Qnvec> QnNdarray;
    // For 3D arrays (e.g., edges: 3 x 2 x 2)
    typedef vector< vector<qnv::Qnvec> > QnNdarray3;

    // Create a 2D array of zeros with given rows and cols.
    inline QnNdarray zeros2D(int rows, int cols) {
        return QnNdarray(rows, qnv::zerov(cols));
    }
    
    // Create a 3D array of zeros with dimensions dim1 x dim2 x dim3.
    inline QnNdarray3 zeros3D(int dim1, int dim2, int dim3) {
        QnNdarray3 arr(dim1);
        for (int i = 0; i < dim1; i++) {
            arr[i] = zeros2D(dim2, dim3);
        }
        return arr;
    }
}

// Namespace for qnmat functions
namespace qnm {
    // Multiply vector v by scalar s.
    inline qnv::Qnvec mul_scl(const qnv::Qnvec &v, double s) {
        qnv::Qnvec res = v;
        for (auto &x : res) {
            x *= s;
        }
        return res;
    }
}

// Namespace for crsys constants
namespace crs {
    const int n = 3;   // Assume 3 for 3D coordinates
    const int ni = 3;  // Typically same as n
}

// Namespace for numeric functions
namespace num {
    // Dummy implementation: check intersection of two segments (returns false)
    inline bool check_intersection_two_segment_numerical_nd_tau(const qna::QnNdarray &segment_1, const qna::QnNdarray &segment_2) {
        return false;
    }
    // Dummy implementation: check intersection of a segment and a surface (returns false)
    inline bool check_intersection_segment_surface_numerical_nd_tau(const qna::QnNdarray &segment, const qna::QnNdarray &surface) {
        return false;
    }
}

// Namespace for prjop (empty as in original code)
namespace prj { }

// Dummy implementations for missing helper functions

// Remove doubling in perpendicular space (dummy: returns input unchanged)
qna::QnNdarray remove_doubling_in_perp_space(const qna::QnNdarray &obj) {
    return obj;
}

// Projection in 3D (dummy: returns input unchanged)
qnv::Qnvec projection3(const qnv::Qnvec &a) {
    return a;
}

// Calculate Euclidean length of a vector
double length_numerical(const qnv::Qnvec &a) {
    double sum = 0.0;
    for (double x : a) {
        sum += x * x;
    }
    return std::sqrt(sum);
}

// Calculate the centroid of a triangle (average of vertices)
qnv::Qnvec centroid(const qna::QnNdarray &tri) {
    qnv::Qnvec cen(crs::n, 0.0);
    if(tri.empty())
        return cen;
    for (auto &v : tri) {
        for (int i = 0; i < crs::n; i++) {
            cen[i] += v[i];
        }
    }
    for (int i = 0; i < crs::n; i++) {
        cen[i] /= tri.size();
    }
    return cen;
}

// Check if a vertex is inside a triangle using barycentric coordinates (assumes 2D triangle)
bool inside_outside_triangle_tau(const qnv::Qnvec &vtx, const qna::QnNdarray &triangle) {
    if(triangle.size() != 3 || vtx.size() < 2)
        return false;
    double x = vtx[0], y = vtx[1];
    double x1 = triangle[0][0], y1 = triangle[0][1];
    double x2 = triangle[1][0], y2 = triangle[1][1];
    double x3 = triangle[2][0], y3 = triangle[2][1];
    double det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
    double a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det;
    double b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det;
    double c = 1 - a - b;
    return (a >= 0 && b >= 0 && c >= 0);
}

// Dummy centroid for an object (returns the object itself)
qnv::Qnvec centroid_obj(const qnv::Qnvec &obj) {
    return obj;
}

// Dummy triangulation (returns the input points)
qna::QnNdarray triangulation_points(const qna::QnNdarray &pts) {
    return pts;
}

// Function: get_edge
// edges in the triangle
qna::QnNdarray3 get_edge(const qna::QnNdarray &tri) {
    // Create a 3 x 2 x 2 array of zeros.
    qna::QnNdarray3 edge = qna::zeros3D(3, 2, 2);
    //printf("triangle.shape in get_edge\n");
    for (int i = 0; i < 3; i++) { // 0 1 2
        int j = (i + 1) % 3;    // 1 2 0
        //printf("i,j %d %d\n", i, j);  // for test
        for (int k = 0; k < 2; k++) {
            edge[i][0][k] = tri[i][k]; // a,b,c
            edge[i][1][k] = tri[j][k]; // b,c,a
        }
        //printf("i %d ", i);  // for test
        // qnv.printqnvs("edge[i]", edge[i]);  // for test
    }
    return edge;
}

// Function: det_vecabc
qnn::Qnnum det_vecabc(const qnv::Qnvec &a, const qnv::Qnvec &b, const qnv::Qnvec &c) {
    qnv::Qnvec da = a;
    qnv::Qnvec db = b;
    // Compute da = a - c and db = b - c
    for (size_t i = 0; i < da.size(); i++) {
        da[i] = a[i] - c[i];
    }
    for (size_t i = 0; i < db.size(); i++) {
        db[i] = b[i] - c[i];
    }
    return da[0] * db[1] - db[0] * da[1];
}

// Function: counter_clockwise
qna::QnNdarray counter_clockwise(qna::QnNdarray tri) {
    if (det_vecabc(tri[0], tri[1], tri[2]) > qnn::zero()) {
        return tri;
    } else { // swap tri[0] and tri[1]
        qnv::Qnvec tmp = qnv::copy(tri[0]);
        tri[0] = tri[1];
        tri[1] = qnv::copy(tmp);
        return tri;
    }
}

// Function: common_points
// common vertices of tri_1 and tri_2 vertices in their intersection
pair<int, qna::QnNdarray> common_points(const qna::QnNdarray &tri_1, const qna::QnNdarray &tri_2) {
    qnv::Qnvec det1 = qnv::zerov(3);
    qnv::Qnvec det2 = qnv::zerov(3);
    qna::QnNdarray comx = qna::zeros2D(6, 2);
    int n = 0;
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < 3; i++) { // 0 1 2
            int j = (i + 1) % 3; // 1 2 0
            det1[i] = det_vecabc(tri_1[i], tri_1[j], tri_2[k]);
        }
        if (det1[0] >= qnn::zero() && det1[1] >= qnn::zero() && det1[2] >= qnn::zero()) {
            // tri_2[k] is in tri_1 or on the border of tri_1
            comx[n] = tri_2[k];
            n += 1;
        }
        for (int i = 0; i < 3; i++) { // 0 1 2
            int j = (i + 1) % 3; // 1 2 0
            det2[i] = det_vecabc(tri_2[i], tri_2[j], tri_1[k]);
        }
        if (det2[0] >= qnn::zero() && det2[1] >= qnn::zero() && det2[2] >= qnn::zero()) {
            // tri_1[k] is in tri_2 or on the border of tri_2
            comx[n] = tri_1[k];
            n += 1;
        }
    }
    // Create sub-array of comx with first n elements
    qna::QnNdarray res(comx.begin(), comx.begin() + n);
    return make_pair(n, res);
}

// Function: rmv_overlapedx
// x : 2D points
pair<int, qna::QnNdarray> rmv_overlapedx(qna::QnNdarray x, int n0) {
    //printf("x.shape\n");  // for test
    int n = 0;
    for (int i = 0; i < n0; i++) {
        if (i == 0) {
            n += 1;
            continue;
        }
        int iskp = 0;
        for (int j = 0; j < n; j++) {
            if (x[i][0] == x[j][0] && x[i][1] == x[j][1]) {
                iskp = 1;
                break;
            }
        }
        if (iskp == 0) {
            x[n] = x[i];
            n += 1;
        }
    }
    qna::QnNdarray res(x.begin(), x.begin() + n);
    return make_pair(n, res);
}

// Function: intersection_two_triangles
// new version: calculates all intersection points of two triangle edges and returns all cross points on the edges
pair<int, qna::QnNdarray> intersection_two_triangles(const qna::QnNdarray &triangle_1, const qna::QnNdarray &triangle_2) {
    //printf("triangle_1.shape\n");  // for test
    //printf("triangle_2.shape\n");  // for test
    //qnv.printqnvs("triangle_1", triangle_1);  // for test
    qna::QnNdarray3 edge_1 = get_edge(triangle_1);
    //printf("edge_1.shape\n");  // for test
    //qnv.printqnvs("triangle_2", triangle_2);  // for test
    qna::QnNdarray3 edge_2 = get_edge(triangle_2);
    //printf("edge_2.shape\n");  // for test
    // cross points of edge_1 and edge_2
    auto t = qna::zeros3D(3, 3, 2);
    qna::QnNdarray x = qna::zeros2D(9, 2);   // cross points
    int n = 0;
    for (int i = 0; i < 3; i++) { // iterate over edge_1 segments
        for (int j = 0; j < 3; j++) { // iterate over edge_2 segments
            // e1 and e2 are line segments (each is 2x2 array)
            vector<qnv::Qnvec> e1 = edge_1[i];
            vector<qnv::Qnvec> e2 = edge_2[j];
            double den = (e1[0][0] - e1[1][0]) * (e2[0][1] - e2[1][1])
                       - (e1[0][1] - e1[1][1]) * (e2[0][0] - e2[1][0]);
            //printf("i,j %d %d ", i, j);  // for test
            //qnn.printqnn("den", den);  // for test
            if (den == qnn::zero()) {  // e1 // e2
                continue;
            }
            // Compute de1 and de2 = differences of endpoints
            qnv::Qnvec de1 = e1[1];
            qnv::Qnvec de2 = e2[1];
            for (size_t k = 0; k < de1.size(); k++) {
                de1[k] = e1[1][k] - e1[0][k];
            }
            for (size_t k = 0; k < de2.size(); k++) {
                de2[k] = e2[1][k] - e2[0][k];
            }
            if (den == qnn::zero()) {  // no cross point (lines are parallel)
                continue;
            }
            double nux = (e1[0][0] - e2[0][0]) * (e2[0][1] - e2[1][1])
                       - (e1[0][1] - e2[0][1]) * (e2[0][0] - e2[1][0]);
            double nuy = (e1[0][0] - e1[1][0]) * (e1[0][1] - e2[0][1])
                       - (e1[0][1] - e1[1][1]) * (e1[0][0] - e2[0][0]);
            t[i][j][0] = nux / den; // t
            t[i][j][1] = -nuy / den; // u
            //qnv.printqnv("t[i][j]", t[i][j]);  // for test
            // if 0<=t[i][j][0]<=1 and 0<=t[i][j][1]<=1 then lines have intersection on i and j-th edges of triangles 1 and 2
            if (t[i][j][0] >= qnn::zero() && t[i][j][0] <= qnn::one() &&
                t[i][j][1] >= qnn::zero() && t[i][j][1] <= qnn::one()) {
                //printf("i,j %d %d ", i, j);  // for test
                //qnn.printqnn("t[i][j][0]", t[i][j][0]);  // for test
                // First calculation (not incrementing n here as in original code commented out)
                qnv::Qnvec pt = qnv::copy(e1[0]);
                for (size_t k = 0; k < pt.size(); k++) {
                    pt[k] += de1[k] * t[i][j][0];
                }
                // Second calculation always applied
                pt = qnv::copy(e2[0]);
                for (size_t k = 0; k < pt.size(); k++) {
                    pt[k] += de2[k] * t[i][j][1];
                }
                x[n] = pt;
                n += 1;
            }
        }
    }
    // Remove overlapped points
    auto rmvRes = rmv_overlapedx(x, n);
    n = rmvRes.first;
    x = rmvRes.second;
    return make_pair(n, x);
}

// Function: common_part
qna::QnNdarray common_part(int nx, const qna::QnNdarray &x, int ny, const qna::QnNdarray &y) {
    cout << "nx " << nx << " ny " << ny << endl;
    qna::QnNdarray z = qna::zeros2D(nx + ny, 2);
    int n = 0;
    for (int i = 0; i < nx; i++) {
        z[n] = x[i];
        n += 1;
    }
    for (int i = 0; i < ny; i++) {
        z[n] = y[i];
        n += 1;
    }
    return z;
}

// Function: ball_radius_obj
// estimate maximum distance between vertices of given OBJ and its centroid.
qnn::Qnnum ball_radius_obj(const qnv::Qnvec &obj, const qnv::Qnvec &centroid) {
    // vertices=remove_doubling_in_perp_space(obj)
    qna::QnNdarray vertices = remove_doubling_in_perp_space(qna::zeros2D(0, 0)); // Dummy: not a real conversion from obj.
    qnv::Qnvec qn0 = {0, 0, 1};
    double dd = qn0[2];  // Using third component as in original assignment dd = qn0
    for (auto &v : vertices) {
        qnv::Qnvec a = v;
        for (size_t i = 0; i < a.size(); i++) {
            a[i] = v[i] - centroid[i];
        }
        a = projection3(a);
        double dd1 = qnv::dot(a, a);
        if (dd1 > dd) {
            dd = dd1;
        } else {
            // pass
        }
    }
    return dd;
}

// Function: ball_radius
qnn::Qnnum ball_radius(const qna::QnNdarray &triangle, const qnv::Qnvec &centroid) {
    // this transforms a tetrahedron to a ball which covers the triangle
    // the centre of the ball is the centroid of the triangle.
    return ball_radius_obj(triangle[0], centroid);
}

// Function: distance_in_perp_space
qnn::Qnnum distance_in_perp_space(const qnv::Qnvec &vt1, const qnv::Qnvec &vt2) {
    qnv::Qnvec a = vt1;
    for (size_t i = 0; i < a.size(); i++) {
        a[i] = vt1[i] - vt2[i];
    }
    a = projection3(a);
    return length_numerical(a);
}

// Function: rough_check_intersection_triangle_obj
bool rough_check_intersection_triangle_obj(const qna::QnNdarray &triangle, const qnv::Qnvec &cententer, qnn::Qnnum distance) {
    qnv::Qnvec cen1 = centroid(triangle);
    qnn::Qnnum dd1 = ball_radius(triangle, cen1);
    qnn::Qnnum dd0 = distance_in_perp_space(cen1, cententer);
    if (dd0 <= dd1 + distance) { // two balls are intersecting.
        return true;
    } else {
        return false;
    }
}

// Function: check_intersection_two_triangles
int check_intersection_two_triangles(const qna::QnNdarray &triangle_1, const qna::QnNdarray &triangle_2) {
    // checking whether triangle_1 is fully inside triangle_2 or not
    int counter2 = 0;
    for (auto &vtx : triangle_1) {
        if (inside_outside_triangle_tau(vtx, triangle_2)) { // inside
            ; // pass
        } else {
            counter2 += 1;
            break;
        }
    }
    // checking whether triangle_2 is fully inside triangle_1 or not
    int counter3 = 0;
    for (auto &vtx : triangle_2) {
        if (inside_outside_triangle_tau(vtx, triangle_1)) { // inside
            ; // pass
        } else {
            counter3 += 1;
            break;
        }
    }
    if (counter2 == 0) {
        return 1; // triangle_1 is fully inside triangle_2
    } else if (counter3 == 0) {
        return 2; // triangle_2 is fully inside triangle_1
    } else {
        //
        // -----------------
        // triangle_1
        // -----------------
        // vertex 1: triangle_1[0],  consist of (a1+b1*TAU)/c1, ... (a6+b6*TAU)/c6    a_i,b_i,c_i = tetrahedron_1[0][i:0~5][0],tetrahedron_1[0][i:0~5][1],tetrahedron_1[0][i:0~5][2]
        // vertex 2: triangle_1[1]
        // vertex 3: triangle_1[2]
        //
        // 1 triangle of triangle_1
        // surface 1: v1,v2,v3
        //
        // 3 edges of triangle_1
        // edge 1: v1,v2
        // edge 2: v1,v3
        // edge 3: v2,v3
        //
        // -----------------
        // triangle_2
        // -----------------
        // vertex 1: triangle_2[0]
        // vertex 2: triangle_2[1]
        // vertex 3: triangle_2[2]
        //
        // 1 surfaces of triangle_2
        // surface 1: w1,w2,w3
        //
        // 3 edges of triangle_2
        // edge 1: w1,w2
        // edge 2: w1,w3
        // edge 3: w2,w3
        //
        // case 1: intersection between (edge of triangle_1) and (surface of triangle_2)
        // case 2: intersection between (edge of triangle_2) and (surface of triangle_1)
        //
        // combination_index
        // e.g. v1,v2,w1,w2,w3 (edge 1 and surface 1) ...
        
        //comb = [ [0,1,0,1,2], [0,2,0,1,2], [1,2,0,1,2] ]
        vector< vector<int> > comb = {
            {0, 1},
            {0, 2},
            {1, 2}
        };
    
        int counter1 = 0;
        for (auto &c : comb) {
            // case 1: intersection between
            // 3 edges of triangle_1
            // 1 surfaces of triangle_2
            qna::QnNdarray segment = { triangle_1[c[0]], triangle_1[c[1]] }; // stack as 2x?
            qna::QnNdarray surface = triangle_2;
            if (num::check_intersection_segment_surface_numerical_nd_tau(segment, surface)) { // intersecting
                counter1 += 1;
                break;
            } else {
                ;
            }
            // case 2: intersection between
            // 3 edges of triangle_2
            // 1 surfaces of triangle_1
            segment = { triangle_2[c[0]], triangle_2[c[1]] };
            surface = triangle_1;
            if (num::check_intersection_segment_surface_numerical_nd_tau(segment, surface)) { // intersecting
                counter1 += 1;
                break;
            } else {
                ;
            }
        }
        if (counter1 > 0) {
            return 3; // intersecting
        } else {
            return 0; // no intersection
        }
    }
}

// Function: intersection_two_segment
// check intersection between two line segments.
qnv::Qnvec intersection_two_segment(const qna::QnNdarray &segment_1, const qna::QnNdarray &segment_2) {
    // check whether two line segments are intersecting or not by numerical calc.
    if (num::check_intersection_two_segment_numerical_nd_tau(segment_1, segment_2)) { // intersecting
        // calc in TAU-style
        qnv::Qnvec vecAB = segment_1[1];
        qnv::Qnvec vecCD = segment_2[1];
        qnv::Qnvec vecAC = segment_2[0];
        for (size_t i = 0; i < vecAB.size(); i++) {
            vecAB[i] = segment_1[1][i] - segment_1[0][i];
        }
        for (size_t i = 0; i < vecCD.size(); i++) {
            vecCD[i] = segment_2[1][i] - segment_2[0][i];
        }
        for (size_t i = 0; i < vecAC.size(); i++) {
            vecAC[i] = segment_2[0][i] - segment_1[0][i];
        }
        double bunbo = qnv::dot(vecAB, vecCD) * qnv::dot(vecCD, vecAB)
                     - qnv::dot(vecAB, vecAB) * qnv::dot(vecCD, vecCD);
        if (bunbo == qnn::zero()) {
            cout << "bunbo=0" << endl;
            return { qnn::inf() };
        }
        double bunshi = qnv::dot(vecAC, vecCD) * qnv::dot(vecCD, vecAB)
                      - qnv::dot(vecCD, vecCD) * qnv::dot(vecAC, vecAB);
        double s = bunshi / bunbo;
        qnn::printqnn("s", s);  // for test
        qnv::Qnvec tmp = qnm::mul_scl(vecAB, s); // vec tunes scale
        qnv::printqnv("tmp", tmp);
        // OP = OA + s*AB
        qnv::Qnvec res = segment_1[0];
        for (size_t i = 0; i < res.size(); i++) {
            res[i] += tmp[i];
        }
        return res;
    } else { // no intersection
        return { qnn::inf() };
    }
}

// Function: intersection_segment_surface
// check intersection between a line segment and a triangle using Möller–Trumbore intersection algorithm
qna::QnNdarray intersection_segment_surface(const qna::QnNdarray &segment, const qna::QnNdarray &surface) {
    // check whether the line segment and the surface are intersecting or not by numerical calc.
    if (num::check_intersection_segment_surface_numerical_nd_tau(segment, surface)) { // intersecting
        /*
        # calc in TAU-style
        vec6AB=sub_vectors(segment[1],segment[0])
        vecAB=projection3(vec6AB)              # AB # R
        #
        tmp=sub_vectors(surface[1],surface[0])
        vecCD=projection3(tmp)                 # CD # E1
        #
        tmp=sub_vectors(surface[2],surface[0])
        vecCE=projection3(tmp)                 # CE # E2
        #
        tmp=sub_vectors(segment[0],surface[0])
        vecCA=projection3(tmp)                 # CA # T
        
        vecP=outer_product(vecAB,vecCE) # P
        vecQ=outer_product(vecCA,vecCD) # Q
        
        bunbo=inner_product(vecP,vecCD)
        
        bunshi=inner_product(vecQ,vecCE)
        t=div(bunshi,bunbo)
        
        # intersecting point: OA + t*AB
        tmp=mul_vector(vec6AB,t) # t*AB
        #printf("   t=",numeric_value(t))
        n=crs.n
        return add_vectors(segment[0],tmp).reshape(1,n)
        */
        //  edge: 0-1,0-2,1-2
        vector< vector<int> > comb = { {0,1}, {0,2}, {1,2} };
        int counter = 0;
        int n = crs::ni;
        qna::QnNdarray p;
        for (auto &j : comb) {
            qna::QnNdarray segment1 = { surface[j[0]], surface[j[1]] }; // stack the points
            cout << "segment.shape " << segment.size() << " segment1.shape " << segment1.size() << endl; // for test
            qnv::Qnvec tmp1 = intersection_two_segment(segment, segment1);
            // if tmp1 equals zerov(n) (dummy check)
            if (tmp1 == qnv::zerov(n)) {
                ; // pass
            } else {
                qnn::printqnn("tmp1", tmp1[0]);
                if (counter == 0) {
                    p.push_back(tmp1);
                } else {
                    // vertical stack: simply push back another intersection point
                    p.push_back(tmp1);
                }
                counter += 1;
            }
        }
        if (counter > 0) { // intersection
            cout << "p.shape " << p.size() << endl;  // for test
            return p;
        } else { // no intersection
            return qna::zeros2D(1, n); // Return zero vector of dimension n
        }
    } else { // no intersection
        int n = crs::ni;
        return qna::zeros2D(1, n);
    }
}

// Function: intersection_two_triangles0
// calculating intersection of triangle_1 and triangle_2 (original version)
qna::QnNdarray intersection_two_triangles0(const qna::QnNdarray &triangle_1, const qna::QnNdarray &triangle_2) {
    //
    // -----------------
    // triangle_1
    // -----------------
    // vertex 1: triangle_1[0],  with nD TAU style coordinates
    // vertex 2: triangle_1[1]
    // vertex 3: triangle_1[2]
    //
    // 2 surface of triangle_1
    // surface 1: v1,v2,v3
    //
    // 3 edges of triangle_1
    // edge 1: v1,v2
    // edge 2: v1,v3
    // edge 3: v2,v3
    //
    // -----------------
    // triangle_2
    // -----------------
    // vertex 1: triangle_2[0]
    // vertex 2: triangle_2[1]
    // vertex 3: triangle_2[2]
    //
    // 2 surfaces of triangle_2
    // surface 1: w1,w2,w3
    //
    // 3 edges of triangle_2
    // edge 1: w1,w2
    // edge 2: w1,w3
    // edge 3: w2,w3
    //
    // case 1: intersection between (edge of triangle_1) and (surface of triangle_2)
    // case 2: intersection between (edge of triangle_2) and (surface of triangle_1)
    //
    // combination_index
    // e.g. v1,v2,w1,w2,w3 (edge 1 and surface 1) ...
    vector< vector<int> > comb = {
        {0, 1, 0, 1, 2},
        {0, 2, 0, 1, 2},
        {1, 2, 0, 1, 2}
    };
    cout << "triangle_1.shape " << triangle_1.size() << " triangle_2.shape " << triangle_2.size() << endl;  // for test
    int counter = 0;
    qna::QnNdarray tmp; // will collect intersection points
    for (auto &c : comb) {
        // case 1: intersection between (edge of triangle_1) and (surface of triangle_2)
        qna::QnNdarray segment = { triangle_1[c[0]], triangle_1[c[1]] };
        qna::QnNdarray surface = { triangle_2[c[2]], triangle_2[c[3]], triangle_2[c[4]] };
        cout << "segment.shape " << segment.size() << " surface.shape " << surface.size() << endl;  // for test
        qna::QnNdarray vtx = intersection_segment_surface(segment, surface);
        cout << "vtx received" << endl;
        // For dummy check, compare with zero vector (dummy implementation)
        if (vtx == qna::zeros2D(1, 1)) {
            ; // pass
        } else {
            if (counter == 0) {
                tmp = vtx; // intersection points
            } else {
                // vertical stacking: append rows
                for (auto &row : vtx) {
                    tmp.push_back(row);
                }
            }
            counter += 1;
        }
        
        // case 2: intersection between (edge of triangle_2) and (surface of triangle_1)
        segment = { triangle_2[c[0]], triangle_2[c[1]] };
        surface = { triangle_1[c[2]], triangle_1[c[3]], triangle_1[c[4]] };
        vtx = intersection_segment_surface(segment, surface);
        if (vtx == qna::zeros2D(1, 1)) {
            ; // pass
        } else {
            if (counter == 0) {
                tmp = vtx; // intersection points
            } else {
                for (auto &row : vtx) {
                    tmp.push_back(row);
                }
            }
            counter += 1;
        }
    }
    int n = crs::ni;  // using crs.ni for dimension
    // The following reshape and further processing is nontrivial.
    // get vertices of triangle_1 inside triangle_2
    for (auto &vtx : triangle_1) {
        if (inside_outside_triangle_tau(vtx, triangle_2)) { // inside
            if (counter == 0) {
                qna::QnNdarray temp;
                temp.push_back(vtx);
                tmp = temp;
            } else {
                tmp.push_back(vtx);
            }
            counter += 1;
        }
    }
    // get vertices of triangle_2 inside triangle_1
    for (auto &vtx : triangle_2) {
        if (inside_outside_triangle_tau(vtx, triangle_1)) { // inside
            if (counter == 0) {
                qna::QnNdarray temp;
                temp.push_back(vtx);
                tmp = temp;
            } else {
                tmp.push_back(vtx);
            }
            counter += 1;
        } else {
            ;
        }
    }
    
    if (counter >= 3) {
        tmp = remove_doubling_in_perp_space(tmp);
        if (tmp.size() > 3) {
            qna::QnNdarray tmp4 = triangulation_points(tmp);
            if (tmp4 == qna::zeros2D(1, 1)) {
                return qna::zeros2D(1, 1);
            } else {
                return tmp4;
            }
        } else if (tmp.size() == 3) {
            // reshape to (1,3,n,3) is not directly representable; return tmp as is.
            return tmp;
        } else {
            return qna::zeros2D(1, 1);
        }
    } else {
        return qna::zeros2D(1, 1);
    }
}

// Function: intersection_two_obj_1
// Return an intersection between two objects.
qnv::Qnvec intersection_two_obj_1(const qnv::Qnvec &obj1, const qnv::Qnvec &obj2, const string &select = "", int verbose = 0) {
    /*
    Return an intersection between two objects.
    
    Parameters
    ----------
    obj1 : ndarray
        a set of triangles to be intersected with obj2.
    obj2 : ndarray
        a set of triangles to be intersected with obj1.
    select : {'standard', 'simple'}, optional
        The default is 'standard'. 
    
    Returns
    -------
    intersection between obj1 and obj2 : ndarray
        Array of the same type and shape as `obj1` and `obj2`.
    
    Notes
    -----
    
    'standard' intersection is default.
    
    Output from 'simple' intersection is simpler but may cause a problem when generating its surface triangles.
    */
    
    if (verbose > 0) {
        cout << "       start: intersection_two_obj_1()" << endl;
    }
    
    qnv::Qnvec cent2 = centroid_obj(obj2);
    qnn::Qnnum dd2 = ball_radius_obj(obj2, cent2);
    
    if (verbose > 0) {
        printf("         dd2:%6.4f\n", dd2);
    }
    
    int counter0 = 0;
    int n = crs::n;
    // The following loop assumes obj1 is a collection of triangles.
    // Since obj1 is of type Qnvec here, this is a dummy implementation.
    // In a complete implementation, obj1 would be a collection (array) of triangles.
    // For demonstration, we leave the loop body incomplete as in the provided code.
    // Loop is truncated in the original code.
    // for (index and triangle1 in obj1) { ... }
    
    // Dummy return value for this incomplete function.
    return cent2;
}

int main() {
    // Example usage (dummy data)
    qna::QnNdarray triangle1 = { {0.0, 0.0}, {1.0, 0.0}, {0.5, 1.0} };
    qna::QnNdarray triangle2 = { {0.5, 0.2}, {1.2, 0.2}, {0.8, 1.1} };
    
    // Test counter_clockwise
    qna::QnNdarray ccw = counter_clockwise(triangle1);
    
    // Test get_edge
    qna::QnNdarray3 edges = get_edge(triangle1);
    
    // Test intersection_two_triangles
    auto interRes = intersection_two_triangles(triangle1, triangle2);
    cout << "Number of intersection points: " << interRes.first << endl;
    for (auto &pt : interRes.second) {
        qnv::printqnv("Intersection point", pt);
    }
    
    // Other function calls can be tested similarly.
    
    return 0;
}
