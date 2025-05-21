#include "geometry.h"
#include <cmath>

point::point(): x(0), y(0) {}
point::point(double x, double y): x(x), y(y) {}

bool point::operator==(const point& other) const {
    if(x == other.x && y == other.y) return true;

    double epsilon = std::numeric_limits<double>::epsilon();
    return fabs(x - other.x) <= epsilon && fabs(y - other.y) <= epsilon;
}

bool point::finite() const {
    double inf = std::numeric_limits<double>::infinity();
    return fabs(x) != inf && fabs(y) != inf;
}

double point::slope(const point& a, const point& b) {
    return a.x - b.x == 0.0 ?
        std::numeric_limits<double>::infinity() :
        (a.y - b.y) / (a.x - b.x);
}

point point::midpoint(const point& a, const point& b) {
    return point((a.x + b.x) / 2.0, (a.y + b.y) / 2.0);
}

double point::distance_squared(const point& a) const {
    double dx = x - a.x;
    double dy = y - a.y;
    return (dx * dx + dy * dy);
}

edge::edge(point a, point b) : a(a), b(b) {}

circle::circle(point center, double radius):
    center(center), radius(radius) {}

bool circle::contains(const point& p) const {
    double dx = (p.x - center.x);
    double dy = (p.y - center.y);
    double dist = dx * dx + dy * dy;

    return dist < radius;
}

bool circle::infinite() const {
    return radius == std::numeric_limits<double>::infinity();
}

triangle::triangle(point a, point b, point c):
    a(a), b(b), c(c) {}

bool triangle::valid() const {
    if(!a.finite() || !b.finite() || !c.finite()) return false;

    /* Compute the triangle validity, e.g., whether the points are collinear.
    ** We can do this by checking the area using the shoelace formula.
    ** The determinant of a matrix with 2 vectors is the area of the
    ** parallelogram spanning these vectors.
    **
    ** 1/2 of this area yields the area of the triangle spanning these vectors.
     */

    // | b.x - a.x  c.x - a.x |
    // | b.y - a.y  c.y - a.y |
    double area = ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / 2.0;

    // Check the actual area, not signed area
    return fabs(area) > std::numeric_limits<double>::epsilon();
}

bool triangle::has_vertex(const point& p) const {
    return (a == p || b == p || c == p);
}

circle triangle::circumcircle() const {
    // Circumcenter can be found by finding the intersection of two
    // perpendicular bisectors

    // Check for collinearity
    if(!valid()) {
        return circle(point(0, 0), std::numeric_limits<double>::infinity());
    }

    double slope_ab = point::slope(a, b);
    double slope_bc = point::slope(b, c);
    double slope_ac = point::slope(a, c);

    point midpoint_1 = point::midpoint(a, b);

    double slope_1 = slope_ab;

    point midpoint_2 = point::midpoint(b, c);

    double slope_2 = slope_bc;

    if(slope_1 == 0.0) {
        midpoint_1 = point::midpoint(a, c);
        slope_1 = slope_ac;
    } else if(slope_2 == 0.0) {
        midpoint_2 = point::midpoint(a, c);
        slope_2 = slope_ac;
    }

    // Calculate the slopes of the perpendicular bisectors
    double m1 = -1/slope_1;
    double m2 = -1/slope_2;

    double b1 = midpoint_1.y - m1 * midpoint_1.x;
    double b2 = midpoint_2.y - m2 * midpoint_2.x;

    /*
    ** Bisectors intersect when y1 = y2,
    **   => m1(x) + b1 = m2(x) + b2
    **   => x(m1 - m2) = b2 - b1
    **   => x = (b2 - b1) / (m1 - m2)
     */
    double x = (b2 - b1) / (m1 - m2);
    double y = m1 * x + b1;

    const point center(x, y);

    /*
    ** While by definition of a circumcircle any point will be on the circle,
    ** due to floating-point precision issues there may be a small difference
    ** from a point to the circle's center.
    **
    ** Due to this, a point on the circumference may be detected as within the
    ** circle. Similar issues still persist when using other formulas for the
    ** radius (Ex: https://mathworld.wolfram.com/Circumradius.html).
    **
    ** Therefore, we're electing to find the smallest radius, such that no
    ** point on the circumference will be detected as within the circle.
     */
    double radius = std::min(
        std::min(a.distance_squared(center), b.distance_squared(center)),
        c.distance_squared(center));

    return circle(center, radius);
}

bool triangle::has_edge(edge e) const {
    std::vector<edge> list = edges();
    for(const edge& v : list) {
        // Undirected edges, so the order does not matter
        if((v.a == e.a && v.b == e.b) || (v.a == e.b && v.b == e.a)) {
            return true;
        }
    }

    return false;
}

std::vector<edge> triangle::edges() const {
    std::vector<edge> result;
    result.emplace_back(a, b);
    result.emplace_back(b, c);
    result.emplace_back(a, c);

    return result;
}

bool triangle::operator==(const triangle& other) const {
    return (a == other.a && b == other.b && c == other.c);
}
