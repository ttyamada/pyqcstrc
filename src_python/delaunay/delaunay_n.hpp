#pragma once
#include <iostream>
#include <limits>    // infinity
#include <algorithm> // infinity
#include "qnnum.hpp"
#include "number.hpp"  // infinity etc.
#include "geometry_n.hpp"

// function template
namespace delaunay {
    //template<class T> T get_inf(T a);
	template<class T> triangle<T> super_triangle();
	template<class T> bool halfplane_contains(triangle<T>& t, point<T>& p);
	template<class T> std::vector<triangle<T>> triangulate(std::vector<point<T>>& points);
};

// implementation
namespace delaunay {

    template<class T>
    triangle<T> super_triangle() {
        //auto inf = std::numeric_limits<T>::infinity(); // only for conventional number
        T a;
        T inf = number::inf<T>();  // geometry::point<T>
        T zero = number::num<T>(0);  // geometry::point<T>
        //std::cout<<"inf "<< inf<<" zero "<<zero<<std::endl;  // for test
        return triangle(point<T>(-inf, -inf), point<T>(zero, inf), point<T>(inf, zero));
    }

    template<class T>
    void wt_triangle(std::string str, triangle<T>& t) {
        std::cout<<str<<t.a.x<<" "<<t.a.y<<" ";  // for test
        std::cout<<t.b.x<<" "<<t.b.y<<" ";  // for test
        std::cout<<t.c.x<<" "<<t.c.y<<std::endl;  // for test
    }

    template<class T>
    bool halfplane_contains(triangle<T>& t, point<T>& p) {
        int finite_points = 0;
        //T inf = std::numeric_limits<T>::infinity();  // only for convensional number
        T a;
        T inf = number::inf<T>();
        finite_points = t.a.finite() + t.b.finite() + t.c.finite();  // 0, 1 or 2

        // Case 3
        if(finite_points == 0) return true;

        if(finite_points == 1) {
            // Case 2: there are 2 infinite vertices
            point<T> f, v1, v2;
            if(t.a.finite()) {
                f = t.a; 
                v1 = t.b;
                v2 = t.c;
            } else if(t.b.finite()) {
                f = t.b;
                v1 = t.a;
                v2 = t.c;
            } else if(t.c.finite()) {
                f = t.c;
                v1 = t.a;
                v2 = t.b;
            }

            if(v1.y == inf || v2.y == inf) {
                if(v1.x == inf || v2.x == inf) {
                    // Vertices: { (0, inf), (inf, 0) }
                    // y = -x + b
                    T b = f.y + f.x;
                    if(p.y + p.x > b) return true;
                } else {
                    // Vertices: { (0, inf), (-inf, -inf) }
                    // y = 3x + b
                    T b = f.y - f.x * 3;
                    if(p.y - p.x * 3 > b) return true;
                }
            } else {
                // Vertices: { (-inf, -inf), (inf, 0) }
                // y = 1/3x + b
                T b = f.y - f.x / 3;
                if(p.y - p.x / 3 < b) return true;
            }
        } else if(finite_points == 2) {
            point<T> f, v1, v2;
            if(!t.a.finite()) {
                f = t.a; v1 = t.b; v2 = t.c;
            } else if(!t.b.finite()) {
                f = t.b; v1 = t.a; v2 = t.c;
            } else if(!t.c.finite()) {
                f = t.c; v1 = t.b; v2 = t.a;
            }

            // The line from v1 to v2 will be tangent to the circle
            T m = point<T>::slope(v1, v2);
            T b = v1.y - m * v1.x;
            if(f.y == inf) {
                // Vertex: (0, inf)
                // The circle interior is always above the line
                if(p.y - m * p.x > b) return true;
            } else if(f.x == inf) {
                // Vertex: (inf, 0)
                if(m >= 0) {
                    if(p.y - m * p.x < b) return true;
                } else if(p.y - m * p.x > b) return true;
            } else {
                // Vertex: (-inf, -inf)
                if(m >= 1) {
                    if(p.y - m * p.x > b) return true;
                } else if(p.y - m * p.x < b) return true;
            }
        }

        return false;
    }
    
    template<class T>
    std::vector<triangle<T>> triangulate(std::vector<point<T>>& points) {

        std::vector<triangle<T>> triangles;

        triangle super = super_triangle<T>();
        wt_triangle<T>("super ",super); // for test
        triangles.push_back(super);

        // Add each point to the triangles
        for(point<T>& p : points) {
            std::cout<<"p "<<p.x<<" "<<p.y<<std::endl;  // for test
            std::vector<triangle<T>> bad_set;

            // Find out which triangles are invalidated when adding this point
            for(triangle<T>& t : triangles) {
                circle circumcircle = t.circumcircle();
                std::cout<<"circumcircle.radius "<<circumcircle.radius<<std::endl;  // for test
                if(!circumcircle.infinite()) {  // circumcircle radius is finite
                    wt_triangle<T>("t ",t); // for test
                    if(circumcircle.contains(p)) bad_set.push_back(t);
                } else {  // circumcircle radius is infinite
                    if(halfplane_contains(t, p)) bad_set.push_back(t);
                }
            }
            std::cout<<"bad_set.size() "<<bad_set.size()<<std::endl;  // for test
            std::vector<edge<T>> polygon;  // polygon edge (double or Qnnum)

            // Find edges not shared with any other flagged triangles
            for(size_t i = 0; i < bad_set.size(); ++i) {
                std::vector<edge<T>> edges = bad_set[i].edges();

                for(edge<T>& e : edges) {
                    bool shared_edge = false;
                    for(size_t j = 0; j < bad_set.size(); ++j) {
                        if(i == j) continue;

                        if(bad_set[j].has_edge(e)) {
                            shared_edge = true;
                            break;
                        }
                    }

                    if(!shared_edge) polygon.push_back(e); // non-shared polygon edges
                }
            }

            // Remove all bad triangles from the triangles
            // definition of predicate only for float or double?
            auto predicate = [&](triangle<T> t) {
                return std::find(bad_set.begin(), bad_set.end(), t) != bad_set.end();  // only for float or double?
            };

            triangles.erase(std::remove_if(triangles.begin(),triangles.end(),predicate), triangles.end());  // only for float or double?

            // Connect edges to our point to form a new triangle
            for(size_t j = 0; j < polygon.size(); ++j) {
                edge e = polygon[j];

                triangles.emplace_back(e.a, e.b, p);
            }
        }

        // Remove any remaining triangle connected to the super triangle
        auto predicate = [&](triangle<T> t) {
            return
                t.has_vertex(super.a) ||
                t.has_vertex(super.b) ||
                t.has_vertex(super.c);
        };

        triangles.erase(
            std::remove_if(triangles.begin(), triangles.end(),
                           predicate), triangles.end());

        return triangles;
    }

    // explicit instantiation
    template triangle<double> super_triangle<double>();  // explicit instantiation
    template bool halfplane_contains<double>(triangle<double>& t, point<double>& p);
    template std::vector<triangle<double>> triangulate<double>(std::vector<point<double>>& points);

    template triangle<qnnum::Qnnum> super_triangle<qnnum::Qnnum>();
    template bool halfplane_contains<qnnum::Qnnum>(triangle<qnnum::Qnnum>& t, point<qnnum::Qnnum>& p);
    template std::vector<triangle<qnnum::Qnnum>> triangulate<qnnum::Qnnum>(std::vector<point<qnnum::Qnnum>>& points);
    
    // for specialization use
    // template<> template triangle<double> super_triangle<double>() {} etc.

}