#pragma once
#include <vector>

class point {
public:
    double x, y; // 2D coordinates
    point();
    point(double x, double y);
    bool operator==(const point& other) const;
    double distance_squared(const point& a) const;
    bool finite() const;
    static double slope(const point& a, const point& b);
    static point midpoint(const point& a, const point& b);
};


class edge {
public:
    point a, b;
    edge(point a, point b);
};

class circle {
public:
    point center;
    double radius;
    circle(point center, double radius);
    bool contains(const point& p) const;
    bool infinite() const;
};

class triangle {
public:
    point a, b, c;
    triangle(point a, point b, point c);
    bool valid() const;
    bool has_vertex(const point&) const;
    circle circumcircle() const;
    bool has_edge(edge e) const;
    std::vector<edge> edges() const;
    bool operator==(const triangle& other) const;
};
