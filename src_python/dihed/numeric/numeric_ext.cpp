// Required dependencies and imports
#include <vector>
#include <cmath>
#include <numeric> // For std::accumulate if needed, or general numeric ops
#include <stdexcept> // For potential error handling
#include <iostream> // For print statements
#include <tuple> // For return type of strc
#include <limits> // For numeric limits if needed
#include <algorithm> // For std::any_of

// Forward declarations for placeholder types and namespaces
namespace qnn {
    // Placeholder for qnn types/functions
    double zero();
}

namespace qnv {
    // Placeholder for Qnvec class
    class Qnvec {
    public:
        std::vector<double> data;
        int N_val = 0; // Placeholder for potential N value associated with vector

        // Default constructor
        Qnvec() = default;

        // Constructor for zero initialization (approximating Python's qnv.Qnvec(n_,N))
        Qnvec(int n, int N) : data(n, 0.0), N_val(N) {}

        // Constructor from std::vector<double>
        Qnvec(const std::vector<double>& d) : data(d) {}
        Qnvec(std::vector<double>&& d) : data(std::move(d)) {}


        // Access operator
        double& operator[](size_t index) {
            if (index >= data.size()) {
                throw std::out_of_range("Qnvec index out of range");
            }
            return data[index];
        }

        const double& operator[](size_t index) const {
            if (index >= data.size()) {
                throw std::out_of_range("Qnvec index out of range");
            }
            return data[index];
        }

        // Get size
        size_t size() const {
            return data.size();
        }

        // Placeholder for slicing operation v[start:end]
        Qnvec slice(size_t start, size_t end) const {
            if (start >= end || end > data.size()) {
                 // Return empty or throw, depending on desired behavior for invalid slice
                 // Python slicing allows start >= end, returning empty.
                 if (start >= data.size() || start >= end) return Qnvec();
                 end = std::min(end, data.size());
                 // throw std::out_of_range("Qnvec slice indices invalid");
            }
            std::vector<double> sliced_data(data.begin() + start, data.begin() + end);
            return Qnvec(std::move(sliced_data));
        }

        // Placeholder for reshape (returns a copy interpreted differently, maybe not needed if functions handle flat data)
        // For simplicity, not implementing complex reshape, assume functions consuming reshaped data handle flat input.

        // Basic arithmetic operators (element-wise)
        Qnvec operator+(const Qnvec& other) const {
            if (size() != other.size()) {
                throw std::invalid_argument("Qnvec sizes must match for addition");
            }
            Qnvec result(size(), this->N_val); // Use N_val of 'this' or decide policy
            for (size_t i = 0; i < size(); ++i) {
                result[i] = data[i] + other.data[i];
            }
            return result;
        }

        Qnvec operator-(const Qnvec& other) const {
            if (size() != other.size()) {
                throw std::invalid_argument("Qnvec sizes must match for subtraction");
            }
            Qnvec result(size(), this->N_val); // Use N_val of 'this' or decide policy
            for (size_t i = 0; i < size(); ++i) {
                result[i] = data[i] - other.data[i];
            }
            return result;
        }
    };
} // namespace qnv

namespace qnm {
    // Placeholder for Qnmat class
    class Qnmat {
    public:
        // Assuming Qnmat might be represented as a vector of vectors or similar
        std::vector<std::vector<double>> matrix_data;

        // Default constructor
        Qnmat() = default;

        // Placeholder method to check if any element is non-zero
        bool any_nonzero() const {
            for (const auto& row : matrix_data) {
                for (double val : row) {
                    if (val != 0.0) {
                        return true;
                    }
                }
            }
            return false;
        }
    };
} // namespace qnm

// Placeholder implementations for functions from imported modules
namespace crs {
    // Placeholder for crsys module elements if needed
}

namespace qnn {
    // Placeholder implementation for qnn.zero()
    double zero() {
        // Returns a value representing zero, potentially with tolerance considerations
        return 0.0; // Or a small epsilon like std::numeric_limits<double>::epsilon()
    }
}

namespace prj {
    // Placeholder implementation for prj.projection3_sets_numerical
    qnv::Qnvec projection3_sets_numerical(const qnv::Qnvec& vts) {
        // --- Placeholder Implementation ---
        // Returns a default-constructed or dummy Qnvec
        // The actual implementation depends entirely on the external library
        std::cout << "Warning: Using placeholder implementation for prj::projection3_sets_numerical" << std::endl;
        return qnv::Qnvec(); // Return a default Qnvec
    }

    // Placeholder implementation for prj.prjvec_e
    qnv::Qnvec prjvec_e(const qnv::Qnvec& vn) {
        // --- Placeholder Implementation ---
        std::cout << "Warning: Using placeholder implementation for prj::prjvec_e" << std::endl;
        return qnv::Qnvec(); // Return a default Qnvec
    }

    // Placeholder implementation for prj.prjop_e (assuming it returns a Qnvec or similar)
    // The Python code `return prj.prjop_e` suggests it might be a pre-calculated object/matrix
    // For now, let's assume it's a function returning a Qnvec for consistency,
    // or define it as a global constant Qnvec if appropriate.
    qnv::Qnvec prjop_e() { // Or potentially: const qnv::Qnvec prjop_e = ...;
        // --- Placeholder Implementation ---
        std::cout << "Warning: Using placeholder implementation for prj::prjop_e" << std::endl;
        return qnv::Qnvec(); // Return a default Qnvec
    }

    // Placeholder for prjvec_i (assuming it's in prj namespace)
     qnv::Qnvec prjvec_i(const qnv::Qnvec& vn) {
        // --- Placeholder Implementation ---
        std::cout << "Warning: Using placeholder implementation for prj::prjvec_i" << std::endl;
        return qnv::Qnvec(); // Return a default Qnvec
    }

} // namespace prj

namespace qna {
    // Placeholder for qnndarray module elements if needed
}

// Placeholder for functions assumed to exist globally or in other modules
// These need proper implementations based on the actual external code.

// Placeholder for projection_numerical
qnv::Qnvec projection_numerical(const qnv::Qnvec& vn) {
    // --- Placeholder Implementation ---
    std::cout << "Warning: Using placeholder implementation for projection_numerical" << std::endl;
    return qnv::Qnvec(); // Return a default Qnvec
}

// Placeholder for projection_numerical_phason
qnv::Qnvec projection_numerical_phason(const qnv::Qnvec& vn, const qnm::Qnmat& pmatrx) {
    // --- Placeholder Implementation ---
    std::cout << "Warning: Using placeholder implementation for projection_numerical_phason" << std::endl;
    return qnv::Qnvec(); // Return a default Qnvec
}

// Placeholder for numerical_vectors
// Assuming it takes a Qnvec representing multiple vectors and returns a std::vector<Qnvec>
std::vector<qnv::Qnvec> numerical_vectors(const qnv::Qnvec& vts) {
     // --- Placeholder Implementation ---
    std::cout << "Warning: Using placeholder implementation for numerical_vectors" << std::endl;
    // Return a vector containing one default Qnvec as a placeholder
    return {qnv::Qnvec()};
}

// Placeholder for numerical_vector
qnv::Qnvec numerical_vector(const qnv::Qnvec& vt) {
     // --- Placeholder Implementation ---
    std::cout << "Warning: Using placeholder implementation for numerical_vector" << std::endl;
    return qnv::Qnvec(); // Return a default Qnvec
}

// Placeholder for triangle_area_numerical
// Assuming it takes a Qnvec representing 3 points (e.g., 9 flat elements for 3x3D points)
double triangle_area_numerical(const qnv::Qnvec& triangle_points) {
    // --- Placeholder Implementation ---
    std::cout << "Warning: Using placeholder implementation for triangle_area_numerical" << std::endl;
    // Requires geometric calculation based on the 3 points encoded in triangle_points
    // Example: If triangle_points contains [x1,y1,z1, x2,y2,z2, x3,y3,z3]
    // Calculate area using cross product or Heron's formula etc.
    if (triangle_points.size() != 9) {
         // Or handle 2D case if size is 6, etc.
         // std::cerr << "Warning: triangle_area_numerical expects 9 elements (3x3D points), got " << triangle_points.size() << std::endl;
         // return 0.0; // Or throw
    }
    // Dummy return value
    return 1.0; // Return a dummy non-zero area
}


// Global constants (from commented-out Python code or inferred)
// const double TAU = std::sqrt(3.0) / 2.0;
// const double SQRT3 = std::sqrt(3.0);
const int N = 3; // Inferred from usage in inout_occupation_domain_numerical

const double EPS = 1e-6; // tolerance (from commented-out Python code, used in inside_outside_triangle_numerical)

// import numpy as np -> Handled by using std::vector or Eigen (here: std::vector in Qnvec)
// import cython -> No direct equivalent needed unless specific Cython features were used. Comment added.
// from numpy.typing import NDArray -> Type hint, replaced by specific C++ types like qnv::Qnvec
// import random -> Not used in the provided snippet

// import crsys as crs -> Placeholder namespace crs
// import qnnum as qnn -> Placeholder namespace qnn
// import qnvec as qnv -> Placeholder namespace qnv and class Qnvec
// import qnmat as qnm -> Placeholder namespace qnm and class Qnmat
// import qnmath as qmt -> No functions used from this in the snippet
// import prjop as prj -> Placeholder namespace prj
// import qnndarray as qna -> Placeholder namespace qna


//TAU=np.sqrt(3)/2.0
//SQRT3=np.sqrt(3)
//N=3

//EPS=1e-6 # tolerance

qnv::Qnvec get_internal_component_sets_numerical(const qnv::Qnvec& vts) {
    """parallel and perpendicular components of a nd lattice vector in direct space.

    Parameters
    ----------
    vsn: array
        set of 6-dimensional vectors, xyzuvw1, xyzuvw2, ...
    """
    //vns=numerical_vectors(vts)
    //return projection3_sets_numerical(vns)
    return prj::projection3_sets_numerical(vts);
}

#########
//  WIP  #
#########
// projection onto Eperp
qnv::Qnvec projection_numerical_perp(const qnv::Qnvec& vn) {
    // Assuming prjvec_i exists in prj namespace based on Python code structure
    return prj::prjvec_i(vn);
    /* This returns nd vector which corresponds to a projection of vn onto Eperp.

    Parameters
    ----------
    v: array
        6-dimensional vector

    Returns
    -------
    nd vectors projected onto Eperp
    */
    //return
}

#########
//  WIP  #
#########
// projection onto Eaparallel
qnv::Qnvec projection_numerical_par(const qnv::Qnvec& vn) {
    return prj::prjvec_e(vn);
    /* This returns nd vector which corresponds to a projection of vn onto Epar.

    Parameters
    ----------
    v: array
        6-dimensional vector

    Returns
    -------
    nd vectors projected onto Eperp. // Note: Comment seems incorrect, should be Epar
    */
    // The Python code had 'return prj.prjop_e' here, which seems inconsistent
    // with the function signature and the previous line 'return prj.prjvec_e(vn)'.
    // Translating the first return statement. If prjop_e was intended,
    // it might be a constant matrix/operator, not a vector projection result.
    // return prj::prjop_e(); // If prjop_e is a function returning Qnvec
}


// Forward declaration needed because inside_outside_triangle_numerical is used by inout_occupation_domain_numerical
bool inside_outside_triangle_numerical(const std::vector<qnv::Qnvec>& triangle, const qnv::Qnvec& point);

// Assuming obj is a collection of triangles, where each triangle is a collection of points (Qnvec)
// Python type hint suggests obj: qnv.Qnvec, but usage obj[i1] and iterating over obj suggests it's a collection.
// Let's assume obj is std::vector<std::vector<qnv::Qnvec>>: vector of triangles, each triangle is vector of points.
// Or maybe obj is std::vector<qnv::Qnvec> where each Qnvec represents a triangle structure?
// The line `triangles[i1]=get_internal_component_sets_numerical(triangle)` suggests `triangle` (iterator variable) is a Qnvec.
// Let's assume obj is std::vector<qnv::Qnvec> where each Qnvec needs processing by get_internal_component_sets_numerical.
// And the result of get_internal_component_sets_numerical is the structure needed by inside_outside_triangle_numerical.
// Let's assume get_internal_component_sets_numerical returns std::vector<qnv::Qnvec> (the 3 vertices).
bool inout_occupation_domain_numerical(const std::vector<qnv::Qnvec>& obj, const qnv::Qnvec& point) {
    """
    """
    int n_ = 3; // Dimension used for temporary Qnvec, assuming 3D internal space
    // qv=qnv.Qnvec(n_,N) # zero initialized qnvec - Not directly used, maybe intended for initialization?
    size_t num = obj.size(); // Get number of triangles/objects from input vector size
    std::vector<std::vector<qnv::Qnvec>> triangles(num); // Store projected triangles (assuming result is vector of points)
    // The Python line `triangles=[qv]*num` is unusual. It creates a list where all elements point to the *same* qv object initially.
    // C++ std::vector<std::vector<qnv::Qnvec>> triangles(num); creates `num` default-constructed inner vectors.

    for (size_t i1 = 0; i1 < num; ++i1) {
        const qnv::Qnvec& triangle_data = obj[i1];
        // Assuming get_internal_component_sets_numerical returns the 3 vertices of the projected triangle
        // And assuming the return type needs conversion or interpretation to fit std::vector<qnv::Qnvec>
        // This part is highly dependent on the actual return type and structure of get_internal_component_sets_numerical
        // For the placeholder, let's assume it returns a Qnvec that needs parsing, or adapt the placeholder.
        // Let's refine the assumption: get_internal_component_sets_numerical returns a Qnvec containing the 3 vertices flattened.
        qnv::Qnvec projected_triangle_flat = get_internal_component_sets_numerical(triangle_data);
        // We need to reconstruct the 3 vertices for inside_outside_triangle_numerical
        // Assuming 3 vertices, each of dimension 3 (based on inside_outside_triangle_numerical usage)
        if (projected_triangle_flat.size() == 9) {
             triangles[i1] = {
                 qnv::Qnvec({projected_triangle_flat[0], projected_triangle_flat[1], projected_triangle_flat[2]}),
                 qnv::Qnvec({projected_triangle_flat[3], projected_triangle_flat[4], projected_triangle_flat[5]}),
                 qnv::Qnvec({projected_triangle_flat[6], projected_triangle_flat[7], projected_triangle_flat[8]})
             };
        } else {
             // Handle error or unexpected size. For placeholder, maybe fill with empty vectors.
             std::cerr << "Warning: Unexpected size from get_internal_component_sets_numerical. Cannot form triangle." << std::endl;
             triangles[i1] = {qnv::Qnvec(), qnv::Qnvec(), qnv::Qnvec()}; // Placeholder empty triangle
        }
    }

    int counter = 0;
    for (const auto& triangle : triangles) {
         // Check if triangle is valid before calling inside_outside
         if (triangle.size() == 3 && triangle[0].size() > 0) { // Basic validity check
            if (inside_outside_triangle_numerical(triangle, point)) { // inside
                counter = 1;
                break;
            }
         }
    }
    if (counter > 0) {
        return true;
    } else {
        return false;
    }
}

// Assuming triangle is std::vector<Qnvec> with 3 elements (vertices)
// Assuming each vertex Qnvec and point Qnvec represent 3D points (size 3)
bool inside_outside_triangle_numerical(const std::vector<qnv::Qnvec>& triangle, const qnv::Qnvec& point) {
    """
    """
    if (triangle.size() != 3 || point.size() != triangle[0].size()) {
        // Basic check for valid input structure
        throw std::invalid_argument("inside_outside_triangle_numerical requires 3 vertices and matching point dimension");
        // Or return false
        // return false;
    }
    // Assuming triangle_area_numerical expects a flat Qnvec of 9 elements [x1,y1,z1, x2,y2,z2, x3,y3,z3]
    // Or adapt if triangle_area_numerical takes std::vector<Qnvec> or similar.

    // Helper lambda to flatten vertices into a Qnvec for triangle_area_numerical
    auto flatten_triangle = [](const qnv::Qnvec& p1, const qnv::Qnvec& p2, const qnv::Qnvec& p3) {
        std::vector<double> flat_data;
        flat_data.insert(flat_data.end(), p1.data.begin(), p1.data.end());
        flat_data.insert(flat_data.end(), p2.data.begin(), p2.data.end());
        flat_data.insert(flat_data.end(), p3.data.begin(), p3.data.end());
        return qnv::Qnvec(flat_data);
    };

    // tmp=np.append(triangle[0],triangle[1])
    // tmp=np.append(tmp,triangle[2])
    // tmp=tmp.reshape(3,3) -> Handled by flatten_triangle and assumption about triangle_area_numerical
    qnv::Qnvec triangle0_flat = flatten_triangle(triangle[0], triangle[1], triangle[2]);
    double area0 = triangle_area_numerical(triangle0_flat);
    //
    // tmp=np.append(point,triangle[1])
    // tmp=np.append(tmp,triangle[2])
    // tmp=tmp.reshape(3,3)
    qnv::Qnvec triangle1_flat = flatten_triangle(point, triangle[1], triangle[2]);
    double area1 = triangle_area_numerical(triangle1_flat);
    //
    // tmp=np.append(point,triangle[0])
    // tmp=np.append(tmp,triangle[2])
    // tmp=tmp.reshape(3,3)
    qnv::Qnvec triangle2_flat = flatten_triangle(point, triangle[0], triangle[2]);
    area1 += triangle_area_numerical(triangle2_flat);
    //
    // tmp=np.append(point,triangle[0])
    // tmp=np.append(tmp,triangle[1])
    // tmp=tmp.reshape(3,3)
    qnv::Qnvec triangle3_flat = flatten_triangle(point, triangle[0], triangle[1]);
    area1 += triangle_area_numerical(triangle3_flat);

    //N=triangle[0].N // N not used here, commented out. Accessing N_val if needed: triangle[0].N_val
    double qn0 = qnn::zero(); // Get the zero value/tolerance from qnn namespace
    // Using EPS directly as per Python comment, or use qn0 if it represents tolerance
    if (std::abs(area0 - area1) < EPS) { // Use EPS constant
        return true; // inside
    } else {
        return false; // outside
    }
}

// structure under linear phason
// Adjusting types based on usage:
// objs: std::vector<std::vector<qnv::Qnvec>> (list of independent domains, each domain is list of equivalent ODs (obj2))
// positions: std::vector<qnv::Qnvec> (list of position vectors corresponding to objs)
// pmatrx: qnm::Qnmat
// eshift: std::vector<qnv::Qnvec>
// oshift: qnv::Qnvec
// Return type: std::vector<std::tuple<qnv::Qnvec, int, int, int, int, int, int>>
std::vector<std::tuple<qnv::Qnvec, int, int, int, int, int, int>> strc(
    const std::vector<std::vector<qnv::Qnvec>>& objs, // List of lists of Qnvec (occupation domains)
    const std::vector<qnv::Qnvec>& positions,       // List of Qnvec (positions)
    const qnm::Qnmat& pmatrx,                       // Phason matrix
    int n1max,
    int n5max,
    const std::vector<qnv::Qnvec>& eshift,          // List of Qnvec (shifts)
    const qnv::Qnvec& oshift,                       // Single Qnvec shift
    int verbose)
{
    """
    """
    std::cout << std::endl;
    std::cout << "len(objs):" << objs.size() << std::endl;
    // Assuming shape means size for Qnvec
    // The Python code iterates through `objs` (list of lists) and prints shape of elements.
    // Let's print the size of the inner lists (number of equivalent domains)
    for (const auto& tmp : objs) {
        std::cout << "tmp.shape (num equivalent domains):" << tmp.size() << std::endl;
        // If you need shape of the Qnvecs inside:
        // if (!tmp.empty()) {
        //     std::cout << "  Qnvec shape (size):" << tmp[0].size() << std::endl;
        // }
    }
    std::cout << "len(positions):" << positions.size() << std::endl;
    for (const auto& tmp : positions) {
        std::cout << "tmp.shape (size):" << tmp.size() << std::endl;
    }


    qnv::Qnvec orgshft;
    int flg = 0;
    // if np.any(pmatrx)!=0:  # under uniform phason strain
    if (pmatrx.any_nonzero()) { // Use placeholder method for Qnmat
        orgshft = projection_numerical_phason(oshift, pmatrx);
        flg = 1;
    } else {
        orgshft = projection_numerical(oshift);
        flg = 0;
    }

    std::vector<std::tuple<qnv::Qnvec, int, int, int, int, int, int>> lst;
    for (int h1 = -n1max; h1 <= n1max; ++h1) {
        if (verbose > 0) {
            std::cout << h1 << std::endl;
        }
        for (int h2 = -n1max; h2 <= n1max; ++h2) {
            for (int h3 = -n1max; h3 <= n1max; ++h3) {
                for (int h4 = -n1max; h4 <= n1max; ++h4) {
                    //for h5 in range(-n5max,n5max+1):
                    for (int h5 = 0; h5 <= n5max; ++h5) { // Python code uses range(0, n5max+1)
                        // vn=np.array([h1,h2,h3,h4,h5,0],dtype=np.float64)
                        qnv::Qnvec vn({(double)h1, (double)h2, (double)h3, (double)h4, (double)h5, 0.0});
                        qnv::Qnvec v;
                        if (flg == 0) {
                            v = projection_numerical(vn);
                        } else {
                            v = projection_numerical_phason(vn, pmatrx);
                        }
                        //-------------------------------------
                        // i-th independent occupation domain
                        //-------------------------------------
                        for (size_t i1 = 0; i1 < objs.size(); ++i1) {
                             const auto& obj1 = objs[i1]; // obj1 is std::vector<qnv::Qnvec>
                             // Assuming positions[i1] is a single Qnvec representing multiple vectors flattened,
                             // or numerical_vectors handles it appropriately.
                             // Let's stick to the placeholder signature: numerical_vectors returns std::vector<Qnvec>
                             std::vector<qnv::Qnvec> pos = numerical_vectors(positions[i1]);
                             qnv::Qnvec xe = numerical_vector(eshift[i1]); // Assuming eshift[i1] is Qnvec
                             std::cout << "   i1:" << i1 << std::endl;
                             // Printing vectors might be verbose, print sizes instead?
                             std::cout << "    pos size:" << pos.size() << std::endl; // Print number of position vectors
                             // for(const auto& p_ : pos) { std::cout << "    pos elem size:" << p_.size() << std::endl; } // Print size of each vector
                             std::cout << "    len(obj1):" << obj1.size() << std::endl;
                             std::cout << "    len(pos):" << pos.size() << std::endl;

                            //print('eshift[i1]:',eshift[i1]) // Requires << operator overload for Qnvec
                            //if flg==0:
                            //    shfte=projection_numerical(xe)
                            //else:
                            //    shfte=projection_numerical_phason(xe,pmatrx)
                            // This calculation seems duplicated later, inside the i2 loop. Following Python structure.

                            // equivalent occupation domains
                            // obj1 is std::vector<qnv::Qnvec> (list of equivalent ODs)
                            for (size_t i2 = 0; i2 < obj1.size(); ++i2) {
                                const qnv::Qnvec& obj2 = obj1[i2]; // obj2 is one OD (Qnvec)
                                // Ensure index i2 is valid for pos vector
                                if (i2 >= pos.size()) {
                                     std::cerr << "Error: Index i2 out of bounds for pos vector." << std::endl;
                                     continue; // Skip this iteration
                                }
                                const qnv::Qnvec& pos_eq = pos[i2];
                                // point=v[3:6]-orgshft[3:6]
                                // Assuming v and orgshft have at least 6 elements
                                if (v.size() < 6 || orgshft.size() < 6) {
                                     std::cerr << "Error: Vector v or orgshft too small for slicing." << std::endl;
                                     continue; // Skip
                                }
                                qnv::Qnvec v_perp = v.slice(3, 6); // Elements 3, 4, 5
                                qnv::Qnvec orgshft_perp = orgshft.slice(3, 6); // Elements 3, 4, 5
                                qnv::Qnvec point = v_perp - orgshft_perp;

                                // Assuming obj2 is a std::vector<qnv::Qnvec> representing the OD structure needed by inout_occupation_domain_numerical
                                // The type of obj2 needs clarification. If obj2 itself IS the OD (e.g., a set of triangles),
                                // then inout_occupation_domain_numerical needs to take obj2 directly.
                                // Let's assume obj2 is std::vector<Qnvec> where each Qnvec is a triangle definition.
                                // This contradicts the earlier assumption that obj1 is std::vector<Qnvec>.
                                // Revisiting Python: `for i2,obj2 in enumerate(obj1):` -> obj1 is iterable, obj2 is an element.
                                // `inout_occupation_domain_numerical(obj2, point)` -> obj2 is passed as the first arg.
                                // Let's assume `obj` in `inout_occupation_domain_numerical` corresponds to `obj2` here.
                                // And `obj2` is of type `std::vector<qnv::Qnvec>` (list of triangles).
                                // This means `objs` must be `std::vector<std::vector<std::vector<qnv::Qnvec>>>`?
                                // Let's stick to the current C++ signature and assume `obj2` (which is Qnvec) needs processing
                                // *before* being passed to inout_occupation_domain_numerical, or that
                                // inout_occupation_domain_numerical can handle a single Qnvec representing the domain.
                                // This ambiguity stems from the lack of definition for the custom types/functions.
                                // Assuming inout_occupation_domain_numerical expects std::vector<Qnvec> as the domain description.
                                // How do we get that from obj2 (which is Qnvec)? This needs clarification.
                                // *Simplifying Assumption for Placeholder*: Let's assume obj2 (a Qnvec) can be directly used
                                // or trivially converted to the format needed by inout_occupation_domain_numerical.
                                // E.g., maybe it contains flattened triangle data.
                                // Let's wrap obj2 in a vector to match the signature of inout_occupation_domain_numerical.
                                std::vector<qnv::Qnvec> obj2_domain = {obj2}; // Wrap obj2 Qnvec into a vector

                                if (inout_occupation_domain_numerical(obj2_domain, point)) { // inside
                                    qnv::Qnvec w;
                                    qnv::Qnvec shfte;
                                    if (flg == 0) {
                                        w = projection_numerical(pos_eq);
                                        shfte = projection_numerical(xe);
                                    } else {
                                        w = projection_numerical_phason(pos_eq, pmatrx);
                                        shfte = projection_numerical_phason(xe, pmatrx);
                                    }
                                    // lst.append([v-w+shfte,i1,h1,h2,h3,h4,h5])
                                    lst.emplace_back(v - w + shfte, static_cast<int>(i1), h1, h2, h3, h4, h5);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return lst;
}
