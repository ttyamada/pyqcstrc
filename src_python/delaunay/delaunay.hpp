#pragma once
#include <iostream>

namespace delaunay {
    triangle super_triangle();
    bool halfplane_contains(const triangle& t, const point& p);
    std::vector<triangle> triangulate(const std::vector<point>& points);
}

// implementation
namespace delaunay {
    triangle super_triangle() {
        /* While we could arbitrarily form a super triangle that encompasses all points,
        ** if three points approach collinearity, their circumcircle will possibly
        ** extend beyond the super triangle if not big enough.
        **
        ** For reference, see post:
        ** https://math.stackexchange.com/questions/4001660
        **
        ** Big shout out to Hagen von Eitzen for suggesting symbolic vertices
        **
        ** So instead, we can just handle the special case where an infinite circle
        ** locally creates a half-plane
        */
        auto inf = std::numeric_limits<double>::infinity();
        return triangle(point(-inf, -inf), point(0.0, inf), point(inf, 0.0));
    }

    bool halfplane_contains(const triangle& t, const point& p) {
        /* If t circumscribes a circle with infinite radius, this circle is
        ** a line locally. We just need to find out which side of the half-plane
        ** a point is in order to tell whether it is interior to the circle or not.
        **
        ** 3 cases to handle:
        ** 3) all three vertices have infinite coordinates:
        **    then our circle simply contains all points
        **
        ** 2) 2 vertices have infinite coordinates,
        **
        ** 1) 1 vertex is infinite
        */

        int finite_points = 0;
        double inf = std::numeric_limits<double>::infinity();

        finite_points = t.a.finite() + t.b.finite() + t.c.finite();

        // Case 3
        if(finite_points == 0) return true;

        if(finite_points == 1) {
            // Case 2: there are 2 infinite vertices
            point f, v1, v2;
            if(t.a.finite()) {
                f = t.a; v1 = t.b; v2 = t.c;
            } else if(t.b.finite()) {
                f = t.b; v1 = t.a; v2 = t.c;
            } else if(t.c.finite()) {
                f = t.c; v1 = t.a; v2 = t.b;
            }

            if(v1.y == inf || v2.y == inf) {
                if(v1.x == inf || v2.x == inf) {
                    // Vertices: { (0, inf), (inf, 0) }
                    // y = -x + b
                    double b = f.y + f.x;
                    if(p.y + p.x > b) return true;
                } else {
                    // Vertices: { (0, inf), (-inf, -inf) }
                    // y = 3x + b
                    double b = f.y - 3.0 * f.x;
                    if(p.y - 3.0 * p.x > b) return true;
                }
            } else {
                // Vertices: { (-inf, -inf), (inf, 0) }
                // y = 1/3x + b
                double b = f.y - f.x / 3.0;
                if(p.y - p.x / 3.0 < b) return true;
            }
        } else if(finite_points == 2) {
            point f, v1, v2;
            if(!t.a.finite()) {
                f = t.a; v1 = t.b; v2 = t.c;
            } else if(!t.b.finite()) {
                f = t.b; v1 = t.a; v2 = t.c;
            } else if(!t.c.finite()) {
                f = t.c; v1 = t.b; v2 = t.a;
            }

            // The line from v1 to v2 will be tangent to the circle
            double m = point::slope(v1, v2);
            double b = v1.y - m * v1.x;
            if(f.y == inf) {
                // Vertex: (0, inf)
                // The circle interior is always above the line
                if(p.y - m * p.x > b) return true;
            } else if(f.x == inf) {
                // Vertex: (inf, 0)
                if(m >= 0.0) {
                    if(p.y - m * p.x < b) return true;
                } else if(p.y - m * p.x > b) return true;
            } else {
                // Vertex: (-inf, -inf)
                if(m >= 1.0) {
                    if(p.y - m * p.x > b) return true;
                } else if(p.y - m * p.x < b) return true;
            }
        }

        return false;
    }

    std::vector<triangle> triangulate(const std::vector<point>& points) {
        /*
        ** Bowyer-Watson algorithm
        ** Reference: https://en.wikipedia.org/wiki/Bowyer-Watson_algorithm
         */
        std::vector<triangle> triangulation;

        triangle super = super_triangle();
        triangulation.push_back(super);

        // Add each point to the triangulation
        for(const point& p : points) {
            std::cout<<p.x<<" "<<p.y<<'\n';
            std::vector<triangle> bad_set;

            // Find out which triangles are invalidated when adding this point
            for(const triangle& t : triangulation) {
                circle circumcircle = t.circumcircle();
                if(!circumcircle.infinite()) {
                    //std::cout<<"circumcircle.infinite() "<<circumcircle.infinite()<<'\n';
                    if(circumcircle.contains(p)) bad_set.push_back(t);
                } else {
                    //std::cout<<"circumcircle.infinite() "<<circumcircle.infinite()<<'\n';
                    if(halfplane_contains(t, p)) bad_set.push_back(t);
                }
            }

            std::vector<edge> polygon;

            // Find edges not shared with any other flagged triangles
            for(size_t i = 0; i < bad_set.size(); ++i) {
                std::vector<edge> edges = bad_set[i].edges();

                for(const edge& e : edges) {
                    bool shared_edge = false;
                    for(size_t j = 0; j < bad_set.size(); ++j) {
                        if(i == j) continue;

                        if(bad_set[j].has_edge(e)) {
                            shared_edge = true;
                            break;
                        }
                    }

                    if(!shared_edge) polygon.push_back(e);
                }
            }


            // Remove all bad triangles from the triangulation
            // definition of predicate
            auto predicate = [&](triangle t) {
                return std::find(bad_set.begin(), bad_set.end(), t) != bad_set.end();
            };

            triangulation.erase(std::remove_if(triangulation.begin(),
                                               triangulation.end(),
                                               predicate), triangulation.end());

            // Connect edges to our point to form a new triangle
            for(size_t j = 0; j < polygon.size(); ++j) {
                edge e = polygon[j];

                triangulation.emplace_back(e.a, e.b, p);
            }
        }

        // Remove any remaining triangle connected to the super triangle
        auto predicate = [&](triangle t) {
            return
                t.has_vertex(super.a) ||
                t.has_vertex(super.b) ||
                t.has_vertex(super.c);
        };

        triangulation.erase(
            std::remove_if(triangulation.begin(), triangulation.end(),
                           predicate), triangulation.end());
        return triangulation;
    }
};
