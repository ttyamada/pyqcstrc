#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

using namespace std;

// --------------------
// Dependencies and Types
// --------------------

// Define qnnum as double
namespace qnn {
    typedef double Qnnum;
    
    inline Qnnum zero() { return 0.0; }
    inline Qnnum one() { return 1.0; }
    inline Qnnum qn2flt(Qnnum t) { return t; }
    inline int any_i(int x) { return x; }
    inline Qnnum abs(Qnnum t) { return std::abs(t); }
    inline void printqnn(const string &label, Qnnum t) {
        cout << label << ": " << t << endl;
    }
}

// Define qnvec as vector of qnnum
namespace qnv {
    typedef vector<qnn::Qnnum> Qnvec;
    
    // Overload addition operator for Qnvec
    inline Qnvec operator+(const Qnvec &a, const Qnvec &b) {
        if(a.size() != b.size())
            throw runtime_error("Size mismatch in operator+ for Qnvec");
        Qnvec result(a.size());
        for (size_t i = 0; i < a.size(); i++)
            result[i] = a[i] + b[i];
        return result;
    }
    
    // Overload subtraction operator for Qnvec
    inline Qnvec operator-(const Qnvec &a, const Qnvec &b) {
        if(a.size() != b.size())
            throw runtime_error("Size mismatch in operator- for Qnvec");
        Qnvec result(a.size());
        for (size_t i = 0; i < a.size(); i++)
            result[i] = a[i] - b[i];
        return result;
    }
    
    // Scalar multiplication
    inline Qnvec operator*(const Qnvec &a, qnn::Qnnum scalar) {
        Qnvec result(a.size());
        for (size_t i = 0; i < a.size(); i++)
            result[i] = a[i] * scalar;
        return result;
    }
    
    // Dot product
    inline qnn::Qnnum dot(const Qnvec &a, const Qnvec &b) {
        if(a.size() != b.size())
            throw runtime_error("Size mismatch in dot for Qnvec");
        qnn::Qnnum sum = qnn::zero();
        for (size_t i = 0; i < a.size(); i++)
            sum += a[i] * b[i];
        return sum;
    }
    
    // Cross product (for 3D vectors)
    inline Qnvec cros(const Qnvec &a, const Qnvec &b) {
        if(a.size() < 3 || b.size() < 3)
            throw runtime_error("Insufficient dimensions for cross product in Qnvec");
        Qnvec result(3);
        result[0] = a[1] * b[2] - a[2] * b[1];
        result[1] = a[2] * b[0] - a[0] * b[2];
        result[2] = a[0] * b[1] - a[1] * b[0];
        return result;
    }
    
    inline void printqnv(const string &label, const Qnvec &v) {
        cout << label << ": [";
        for (size_t i = 0; i < v.size(); i++){
            cout << v[i];
            if(i < v.size() - 1) cout << ", ";
        }
        cout << "]" << endl;
    }
}

// Define qnmat as a 2D matrix of qnnum (vector of Qnvec)
namespace qnm {
    typedef vector<qnv::Qnvec> Qnmat;
    
    // Create a 2D matrix from two Qnvec (concatenating rows)
    inline Qnmat matrix_2d(const qnv::Qnvec &v1, const qnv::Qnvec &v2) {
        Qnmat m;
        m.push_back(v1);
        m.push_back(v2);
        return m;
    }
}

// qnmath functions
namespace qmt {
    // Determinant of a 2x2 matrix; if a dimension parameter is given, use that.
    inline qnn::Qnnum det_matrix(const qnm::Qnmat &m, int dim = 0) {
        // Assume matrix m has at least two rows and two columns.
        if(m.size() >= 2 && m[0].size() >= 2 && m[1].size() >= 2) {
            return m[0][0] * m[1][1] - m[0][1] * m[1][0];
        }
        return qnn::zero();
    }
    
    inline qnn::Qnnum abs(qnn::Qnnum t) {
        return std::abs(t);
    }
}

// Define projection operations in namespace prj
namespace prj {
    // Dummy implementation of prjop_i: returns its argument unchanged.
    inline qnv::Qnvec prjop_i(const qnv::Qnvec &vn) {
        return vn;
    }
    
    // Dummy implementation of prjvec_i: returns first element (or zero if empty)
    inline qnn::Qnnum prjvec_i(const qnv::Qnvec &vn) {
        if(vn.size() > 0)
            return vn[0];
        return qnn::zero();
    }
}

// Define qnndarray as a wrapper for multi–dimensional arrays of qnnum.
// Here we implement a simple 1D container of Qnvec for demonstration.
namespace qna {
    struct QnNdarray {
        vector<qnv::Qnvec> data;
        // Shape stored as vector of sizes (for example: {number_of_elements})
        vector<size_t> shape;
        
        QnNdarray() {}
        QnNdarray(const vector<size_t>& shp) : shape(shp) {
            data.resize(shp[0]);
        }
        
        qnv::Qnvec& operator[](size_t i) { return data[i]; }
        const qnv::Qnvec& operator[](size_t i) const { return data[i]; }
    };
    
    // Create a QnNdarray filled with zeros given a shape.
    inline QnNdarray zeros(const vector<size_t>& shp) {
        QnNdarray arr(shp);
        for (size_t i = 0; i < shp[0]; i++) {
            // For simplicity, initialize each Qnvec as empty.
            arr.data[i] = qnv::Qnvec();
        }
        return arr;
    }
}

// --------------------
// Global Variables (from crsys module and numeric_init)
// --------------------
namespace crs {
    int n = 0;
    int N = 0;
    int isys = 0;
}

qnn::Qnnum n;
qnn::Qnnum N;
qnn::Qnnum isys;

// numeric_init: initializes global variables from crs
void numeric_init() {
    n = crs::n;
    N = crs::N;
    isys = crs::isys;
}

// --------------------
// Function Definitions (line-by-line translation)
// --------------------

// coplanar_check_numeric_tau:
// check the points (pts) are in coplanar or not
bool coplanar_check_numeric_tau(const qnv::Qnvec &pts, int num_iteration = 5) {
    // In the original code, p is obtained by:
    //    p = get_internal_component_sets_numerical(pts)
    // Here we simulate it by wrapping pts into a QnNdarray.
    qna::QnNdarray p;
    p.shape = {1};
    p.data.push_back(pts);
    // Then, return coplanar_check_numeric(p, num_iteration)
    // Since coplanar_check_numeric is not defined in the snippet, we return false as a dummy.
    return false;
}

// dot: computes the dot product between two qnvec vectors.
qnn::Qnnum dot(const qnv::Qnvec &v1, const qnv::Qnvec &v2) {
    size_t len = v1.size();
    qnn::Qnnum v = qnn::zero();
    if (isys == 3) {  // decagonal case
        // In the original code, s2 = prj.scly2 (a qnnumber) is used.
        // Here we set s2 to 1.0 as a dummy value.
        qnn::Qnnum s2 = 1.0;
        v = v1[0] * v2[0] + v1[1] * v2[1] * s2;
        // Note: the commented-out terms are omitted.
    } else {
        for (size_t i = 0; i < len; i++) {
            v = v + v1[i] * v2[i];
        }
    }
    return v;
}

// point_on_segment: judge whether a point is on a line segment (A-B)
bool point_on_segment(const qnv::Qnvec &point, const vector<qnv::Qnvec> &line_segment) {
    // The line_segment is expected to have two elements: start point and end point.
    qnv::Qnvec xyx0 = point;
    qnv::Qnvec xyx1 = line_segment[0];  // start point
    qnv::Qnvec xyx2 = line_segment[1];  // end point
    
    qnv::Qnvec vecPA = xyx0 - xyx1;
    qnv::Qnvec vecBA = xyx2 - xyx1;
    qnn::Qnnum lPA = qnv::dot(vecPA, vecPA);  // squared norm
    qnn::Qnnum lBA = qnv::dot(vecBA, vecBA);  // squared norm
    qnn::Qnnum qn1 = qnn::one();
    qnn::Qnnum qn0 = qnn::zero();
    // if lBA > 0 and abs(dot(vecPA,vecBA)-lPA*lBA)==0:
    if(lBA > qnn::zero() && std::abs(qnv::dot(vecPA, vecBA) - lPA * lBA) == qn0) {
        qnn::Qnnum s = lPA / lBA;
        if(s >= qn0 && s <= qn1) {
            return true;
        } else if(s > qn1) {
            return false;
        } else {
            return false;
        }
    }
    else {
        return false;
    }
}

// on_out_surface: check whether the point is inside the triangle.
bool on_out_surface(const qnv::Qnvec &point, const vector<qnv::Qnvec> &triangle) {
    // Define inner function 'func'
    auto func = [](const qnv::Qnvec &p_xyz, const vector<qnv::Qnvec> &tr_xyz, int indx) -> vector<qnv::Qnvec> {
        vector<qnv::Qnvec> out(3);
        for (int i = 0; i < 3; i++) {
            if(i == indx)
                out[i] = p_xyz;
            else
                out[i] = tr_xyz[i];
        }
        return out;
    };
    
    qnv::Qnvec p = { prj::prjvec_i(point) }; // projection3_numerical returns float; wrap in Qnvec.
    // area0 = area of the original triangle
    qnn::Qnnum area0 = triangle_area_numerical(triangle);
    
    vector<qnv::Qnvec> triangle1 = func(p, triangle, 0);
    vector<qnv::Qnvec> triangle2 = func(p, triangle, 1);
    vector<qnv::Qnvec> triangle3 = func(p, triangle, 2);
    
    qnn::Qnnum area1 = triangle_area_numerical(triangle1) +
                         triangle_area_numerical(triangle2) +
                         triangle_area_numerical(triangle3);
    qnn::Qnnum qn0 = qnn::zero();
    if (std::abs(area0 - area1) == qn0)
        return true;
    else
        return false;
}

// numeric_value: converts a TAU-style value to a float.
double numeric_value(qnn::Qnnum t) {
    return qnn::qn2flt(t);
    /* Unreachable code (original commented-out parts):
       return (t.n[0]+t[1]*TAU)/t.n[2]
       return (t.n[0]+t[1]*SQRT3)/t.n[2]
       return (t.n[0]+t[1]*sqrt(crs.N))/t.n[2]
    */
}

// numerical_vector: converts a TAU-style vector to a numeric vector (float values).
vector<double> numerical_vector(const qnv::Qnvec &vt) {
    // Direct conversion of qnv::Qnvec to vector<double>
    vector<double> result;
    for(auto val: vt)
        result.push_back(val);
    return result;
}

// numerical_vectors: converts TAU-style vectors to float (for 3D or 4D arrays)
vector<vector<double>> numerical_vectors(const qna::QnNdarray &vts) {
    // If vts.ndim==3 (simulated by shape.size()==1)
    if(vts.shape.size() == 1) { // triangle vertex case
        size_t n1 = vts.shape[0];
        vector<vector<double>> w(n1);
        for (size_t i1 = 0; i1 < n1; i1++) {
            w[i1] = numerical_vector(vts.data[i1]);
        }
        return w;
    }
    // Else if vts.ndim==4 (simulate as 2D)
    else if(vts.shape.size() == 2) { // tetrahedron vertex case
        size_t n1 = vts.shape[0], n2 = vts.shape[1];
        vector<vector<double>> w(n1, vector<double>(n2, 0.0));
        for (size_t i1 = 0; i1 < n1; i1++) {
            w[i1] = numerical_vector(vts.data[i1]);
        }
        return w;
    }
    else {
        cout << "error" << endl;
        return vector<vector<double>>();
    }
}

// length_numerical: computes the Euclidean norm of a TAU-style vector.
double length_numerical(const qnv::Qnvec &vt) {
    vector<double> vn = numerical_vector(vt);
    double sum = 0.0;
    for(auto val: vn)
        sum += val * val;
    return sqrt(sum);
}

// check_intersection_segment_surface_numerical_nd_tau:
// check intersection between a line segment and a triangle (nd TAU-style)
bool check_intersection_segment_surface_numerical_nd_tau(const vector<qnv::Qnvec> &line_segment, const vector<qnv::Qnvec> &triangle) {
    // Original code converts to internal component sets then calls check_intersection_segment_surface_numerical.
    return check_intersection_segment_surface_numerical(line_segment, triangle);
}

// check_intersection_segment_surface_numerical:
// check intersection between a line segment and a triangle.
bool check_intersection_segment_surface_numerical(const vector<qnv::Qnvec> &line_segment, const vector<qnv::Qnvec> &triangle) {
    // Consider edges: 0-1, 0-2, 1-2
    vector<vector<int>> comb = { {0,1}, {0,2}, {1,2} };
    int counter = 0;
    for(auto &j : comb) {
        // Create a segment from the triangle edge using indices
        vector<qnv::Qnvec> seg = { triangle[j[0]], triangle[j[1]] };
        if(check_intersection_two_segment_numerical(line_segment, seg)) {
            counter++;
            break;
        }
    }
    if (counter > 0)
        return true;
    else
        return false;
}

// check_intersection_two_segment_numerical_nd_tau:
// check intersection between two line segments (nd TAU-style)
bool check_intersection_two_segment_numerical_nd_tau(const vector<qnv::Qnvec> &segment_1, const vector<qnv::Qnvec> &segment_2) {
    return check_intersection_two_segment_numerical(segment_1, segment_2);
}

// check_intersection_two_segment_numerical:
// check intersection between two 3D line segments.
bool check_intersection_two_segment_numerical(const vector<qnv::Qnvec> &ln1, const vector<qnv::Qnvec> &ln2) {
    // In the original code, several commented-out implementations exist.
    qnv::Qnvec vecAB = ln1[1] - ln1[0]; // edge vector 1
    qnv::Qnvec vecAC = ln2[1] - ln2[0]; // edge vector 2
    qnv::Qnvec vecCD = ln2[1] - ln1[0]; // edge vector 3
    
    // If the vectors have 2 or fewer components, assume the segments are co–planar.
    if(vecAB.size() <= 2) {
        return true;
    }
    // For ndim == 3, check if the segments are on the same plane by computing a determinant.
    qnm::Qnmat m = qnm::matrix_2d(vecAB, vecAC);
    qnn::Qnnum vol = qmt::det_matrix(m);
    if(vol == 0)
        return true;
    else
        return false;
}

// triangle_area:
// Numerical calculation of the area of a given triangle (TAU-style).
qnn::Qnnum triangle_area(const vector<qnv::Qnvec> &a) {
    qnv::Qnvec v1 = a[1] - a[0];
    qnv::Qnvec v2 = a[2] - a[0];
    qnm::Qnmat m = qnm::matrix_2d(v1, v2);
    qnn::Qnnum vol = qmt::det_matrix(m);
    return std::abs(vol) / 2;
}

// triangle_area_numerical:
// Numerical calculation of the area of a triangle.
qnn::Qnnum triangle_area_numerical(const vector<qnv::Qnvec> &a) {
    cout << "a.shape " << a.size() << endl;
    qnv::Qnvec v1 = a[1] - a[0];  // edge vector
    qnv::Qnvec v2 = a[2] - a[0];  // edge vector
    qnm::Qnmat m = qnm::matrix_2d(v1, v2);
    qnn::Qnnum vol = qmt::det_matrix(m, 2);
    qnn::printqnn("vol", vol);
    qnn::Qnnum qn2 = qnn::any_i(2);
    return qmt::abs(vol) / qn2;
}

// inside_outside_obj_tau:
// Judges whether the point is inside an object (set of triangles) (TAU-style).
bool inside_outside_obj_tau(const qnv::Qnvec &point, const vector<qnv::Qnvec> &obj) {
    // TAU-style conversion is assumed to be done internally.
    return inside_outside_obj(point, obj);
}

// inside_outside_obj:
// Judges whether the point is inside an object (set of triangles).
bool inside_outside_obj(const qnv::Qnvec &point, const vector<qnv::Qnvec> &obj) {
    int flg = 0;
    // In the original code, 'obj' is iterated as a set of tetrahedrons; here we assume it is a set of triangles.
    for(auto &triangle : obj) {
        // Call inside_outside_triangle with the given point and a triangle.
        // Since the inside_outside_triangle function below expects a triangle
        // (here we wrap the single triangle in a vector for demonstration),
        if (inside_outside_triangle(point, {triangle})) {
            flg += 1;
            break;
        }
    }
    if(flg == 0)
        return true;  // inside
    else
        return false; // outside
}

// inside_outside_triangle_tau:
// Judges whether the point is inside a triangle (TAU-style).
bool inside_outside_triangle_tau(const qnv::Qnvec &point, const qna::QnNdarray &triangle) {
    // Conversion (if needed) is done; here we pass triangle.data.
    return inside_outside_triangle(point, triangle.data);
}

// small_triangle:
// Replaces one of the triangle vertices with the point.
vector<qnv::Qnvec> small_triangle(int indx, const qnv::Qnvec &point, const vector<qnv::Qnvec> &triangle0) {
    vector<qnv::Qnvec> tri = triangle0; // copy original triangle
    for (size_t i = 0; i < triangle0.size(); i++) {
        if(i == (size_t)indx)
            tri[i] = point;
        else
            tri[i] = triangle0[i];
    }
    return tri;
}

// inside_outside_triangle:
// Judges whether the point is inside a triangle.
bool inside_outside_triangle(const qnv::Qnvec &point, const vector<qnv::Qnvec> &tri) {
    cout << "tri.shape " << tri.size() << endl;
    cout << "point.shape " << point.size() << endl;
    if(tri.size() == 0 || tri.size() > 1) {
        cout << "tri in inside_outside_triangle should be one" << endl;
        exit(1);
    }
    qnn::Qnnum area0 = triangle_area_numerical(tri);
    
    vector<qnv::Qnvec> tet1 = small_triangle(0, point, tri);
    qnn::Qnnum area1 = triangle_area_numerical(tet1);
    
    vector<qnv::Qnvec> tet2 = small_triangle(1, point, tri);
    area1 += triangle_area_numerical(tet2);
    
    vector<qnv::Qnvec> tet3 = small_triangle(2, point, tri);
    area1 += triangle_area_numerical(tet3);
    
    // Compare areas
    if(area0 == area1)
        return true;  // inside
    else
        return false; // outside
}

// obj_volume_nd_numerical:
// Returns volume of an object (set of triangles).
qnn::Qnnum obj_volume_nd_numerical(const vector< vector<qnv::Qnvec> > &obj) {
    qnn::Qnnum qn0 = qnn::zero();
    qnn::Qnnum vol = qn0;
    for(auto &triangle : obj) {
        vol += triangle_volume_nd_numerical(triangle[0]);
    }
    return vol;
}

// triangle_volume_nd_numerical:
// Returns volume of a triangle.
// (In this implementation, volume is equivalent to the area of the triangle.)
qnn::Qnnum triangle_volume_nd_numerical(const qnv::Qnvec &triangle) {
    return triangle_area_numerical({triangle});
}

// get_internal_component_sets_numerical:
// Returns the internal component sets (alias to projection3_sets_numerical)
vector<qnv::Qnvec> get_internal_component_sets_numerical(const qnv::Qnvec &vns) {
    return projection3_sets_numerical({vns}).empty() ? vector<qnv::Qnvec>() : projection3_sets_numerical({vns});
}

// projection_numerical:
// Returns the projection (parallel and perpendicular components) of a vector.
qnv::Qnvec projection_numerical(const qnv::Qnvec &vn) {
    return prj::prjop_i(vn);
}

// projection_sets_numerical:
// Returns projections for a set of vectors.
vector<qnv::Qnvec> projection_sets_numerical(const vector<qnv::Qnvec> &vns) {
    size_t num = vns.size();
    vector<qnv::Qnvec> m;
    for (size_t i = 0; i < num; i++) {
        m.push_back(prj::prjop_i(vns[i]));
    }
    return m;
}

// projection3_numerical:
// Returns the projection (as a float) of a vector (using prj.prjvec_i).
qnn::Qnnum projection3_numerical(const qnv::Qnvec &vn) {
    return prj::prjvec_i(vn);
}

// projection3_sets_numerical:
// Returns projections for a set of vectors, converting to internal space.
vector<qnv::Qnvec> projection3_sets_numerical(const qna::QnNdarray &vns) {
    vector<qnv::Qnvec> m;
    size_t nc = vns.data.size();
    int isys_local = crs::isys;
    int ni = (isys_local > 2) ? 2 : 3;
    for (size_t i = 0; i < nc; i++) {
        qnv::Qnvec projVec;
        qnn::Qnnum projVal = projection3_numerical(vns.data[i]);
        projVec.resize(ni, projVal);
        m.push_back(projVec);
    }
    return m;
}

// projection_numerical_phason:
// Parallel phason projection (dummy implementation using prjop_i).
qnv::Qnvec projection_numerical_phason(const qnv::Qnvec &vn, const qnm::Qnmat &mat) {
    return prj::prjop_i(vn);
}

// --------------------
// Main Function (for Testing)
// --------------------
int main() {
    numeric_init();
    cout << "numeric_init done." << endl;
    cout << "n = " << n << ", N = " << N << ", isys = " << isys << endl;
    // Further testing code can be placed here.
    return 0;
}
  
