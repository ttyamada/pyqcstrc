#pragma once
#include <cmath>
#include <vector>
#include <typeinfo>
#include "qnnum.hpp"
#include "number.hpp"

 // template version for using qnnumber points (T=Qnnum)
template<class T>
class point {
public:
    T x, y;
    point();
    point(T x, T y);  // qnnum coordinates
    bool operator==(const point<T>& other) const ;
    T distance_squared( point<T>& a) const ;
    bool finite() const;
    static T slope(const point<T>& a, const point<T>& b);
    static point<T> midpoint(const point<T>& a,const  point<T>& b);
};

template<class T>
class edge {
public:
    point<T> a, b;
    edge(point<T> a, point<T> b);
};

template<class T>
class circle {
public:
    point<T> center;
    T radius;
    circle(point<T> center, T radius);
    bool contains( point<T>& p) ;
    bool infinite() ;
};

template<class T>
class triangle {
public:
    point<T> a, b, c;
    triangle();  // default constructor
    triangle(point<T> a, point<T> b, point<T> c); // constructor
    bool valid() const ;
    bool has_vertex( point<T>&) const ;
    circle<T> circumcircle() const ;
    bool has_edge(edge<T> e) const ;
    std::vector<edge<T>> edges() const ;
    bool operator==( const triangle<T>& other);
};

template<class T>
T abs(T a) {
   return std::abs(a);  // conventional number
}

// spetialization for qnnum::Qnnum
template<> qnnum::Qnnum abs<qnnum::Qnnum>(qnnum::Qnnum a) {
    return qnnum::abs(a);
}

