#include "geometry_n.hpp"

// implementation of point<T>::point, edge<T>::edge, triangle<T>::triangle classes 
template<class T>
point<T>::point() {
    T a;
    T zero = number::num<T>(0);
    x = zero; y = zero;
}  //constructor

template<class T>
point<T>::point(T x_, T y_): x(x_), y(y_) {}  // constructor

template<class T>
point<T> point<T>::copy(const point<T>& b) {
    return point<T>(b.x, b.y);
}

template<class T>
point<T> point<T>::operator+(const point<T>& b) const {
    point<T> a=*this;
    point<T> c;
    c.x=a.x+b.x;
    c.y=a.y+b.y;
    return point<T>(c.x, c.y);
}

template<class T>
point<T> point<T>::operator-(const point<T>& b) const {
    point<T> a=*this;
    point<T> c;
    c.x=a.x-b.x;
    c.y=a.y-b.y;
    return point<T>(c.x, c.y);
}

template<class T>
bool point<T>::operator==(const point<T>& other) const {
    // point a=*this; a.x a.y
    if(this->x == other.x && this->y == other.y) return true;

    //T epsilon = std::numeric_limits<T>::epsilon();
    T a;
    T epsilon = number::eps<T>();
    return abs<T>(this->x - other.x) < epsilon && abs<T>(this->y - other.y) < epsilon;
}

template<class T>
bool point<T>::finite() const {
    // point a=*this; a.x a.y
    //T inf = std::numeric_limits<T>::infinity();
    T inf = number::inf<T>();
    return abs<T>(this->x) != inf && abs<T>(this->y) != inf;
}

template<class T>
T point<T>::slope(const point<T>& a,const point<T>& b) {
    T a_;
    //return a.x - b.x == number::zero<T>() ? number::inf<T>() :
    //    (a.y - b.y) / (a.x - b.x);
    return a.x - b.x == number::num<T>(0) ? number::inf<T>() :
        (a.y - b.y) / (a.x - b.x);
    //return a.x - b.x == number::num<T>(0) ? std::numeric_limits<T>::infinity() :
    //    (a.y - b.y) / (a.x - b.x);
}

template<class T>
point<T> point<T>::midpoint(const point<T>& a,const point<T>& b) {
    T a_;
    return point((a.x + b.x) / number::num<T>(2), (a.y + b.y) / number::num<T>(2));
}

template<class T>
T point<T>::distance_squared(point<T>& a) const {
    T dx = x - a.x;
    T dy = y - a.y;
    return (dx * dx + dy * dy);
}

template<class T>
edge<T>::edge(point<T> a, point<T> b) : a(a), b(b) {}

template<class T>
circle<T>::circle(point<T> center, T radius):
    center(center), radius(radius) {}

template<class T>
bool circle<T>::contains(point<T>& p) {
    T dx = (p.x - center.x);
    T dy = (p.y - center.y);
    T dist = dx * dx + dy * dy;

    return dist < radius;
}

template<class T>
bool circle<T>::infinite() {    
    return radius == number::inf<T>();
    //return radius == std::numeric_limits<T>::infinity();
}

template<class T>
triangle<T>::triangle() {
    a = point<T>(); b = point<T>();  c = point<T>();}  // constructor

template<class T>
triangle<T>::triangle(point<T> a, point<T> b, point<T> c):
    a(a), b(b), c(c) {}  // constructor

template<class T>
bool triangle<T>::valid() const {
    //point a = this->a; point b=this->b; point c=this->c;
    if(!this->a.finite() || !this->b.finite() || !this->c.finite()) return false;

    /* Compute the triangle validity, e.g., whether the points are collinear.
    ** We can do this by checking the area using the shoelace formula.
    ** The determinant of a matrix with 2 vectors is the area of the
    ** parallelogram spanning these vectors.
    **
    ** 1/2 of this area yields the area of the triangle spanning these vectors.
     */
    // area is given by the determinant
    // | b.x - a.x  c.x - a.x |
    // | b.y - a.y  c.y - a.y |
    T a_;
    T area = ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / number::num<T>(2);

    // Check the actual area, not signed area
    T epsilon = number::eps<T>();
    return abs<T>(area) > epsilon;  //std::numeric_limits<T>::epsilon();
}

template<class T>
bool triangle<T>::has_vertex(point<T>& p) const {
    // point a=this->a; point b=this->b; pointc=this->c
    return (this->a == p || this->b == p || this->c == p);
}

template<class T>
circle<T> triangle<T>::circumcircle() const {
    //point a=this->a; point b=this->b; point c=this->c;
    // Circumcenter can be found by finding the intersection of two
    // perpendicular bisectors

    // Check for collinearity   T zero = number::get_zerp(a);
    T a_;
    T zero = number::num<T>(0);  // number.
    T one = number::num<T>(1);   // number.
    if(!valid()) {
        //return circle(point<T>(zero, zero), std::numeric_limits<T>::infinity());
        return circle(point<T>(zero, zero), number::inf<T>());
    }

    T slope_ab = point<T>::slope(this->a, this->b);
    T slope_bc = point<T>::slope(this->b, this->c);
    T slope_ac = point<T>::slope(this->a, this->c);

    point<T> midpoint_1 = point<T>::midpoint(this->a, this->b);

    T slope_1 = slope_ab;

    point<T> midpoint_2 = point<T>::midpoint(this->b, this->c);

    T slope_2 = slope_bc;
 
    if(slope_1 == zero) {
        midpoint_1 = point<T>::midpoint(this->a, this->c);
        slope_1 = slope_ac;
    } else if(slope_2 == zero) {
        midpoint_2 = point<T>::midpoint(this->a, this->c);
        slope_2 = slope_ac;
    }

    // Calculate the slopes of the perpendicular bisectors
    T m1 = -one/slope_1;
    T m2 = -one/slope_2;

    T b1 = midpoint_1.y - m1 * midpoint_1.x;
    T b2 = midpoint_2.y - m2 * midpoint_2.x;

    /*
    ** Bisectors intersect when y1 = y2,
    **   => m1(x) + b1 = m2(x) + b2
    **   => x(m1 - m2) = b2 - b1
    **   => x = (b2 - b1) / (m1 - m2)
     */
    if ((m1 - m2) == number::zero<T>()) {
        std::cout<<"m1 "<<m1<<" m2 "<<m2<<" m1-m2"<< (m1-m2)<<std::endl;  // for test
    }
    T x = (b2 - b1) / (m1 - m2);
    T y = m1 * x + b1;

    point<T> center(x, y); // constructor

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
    T radius = std::min(
        std::min(a.distance_squared(center), b.distance_squared(center)),
        c.distance_squared(center)); // this works for qnnum.Qnnum?

    return circle(center, radius);
}

template<class T>
bool triangle<T>::has_edge(edge<T> e) const {
    std::vector<edge<T>> list = edges();
    for(edge<T>& v : list) {
        // Undirected edges, so the order does not matter
        if((v.a == e.a && v.b == e.b) || (v.a == e.b && v.b == e.a)) {
            return true;
        }
    }

    return false;
}

template<class T>
std::vector<edge<T>> triangle<T>::edges() const {
    std::vector<edge<T>> result;
    //point a=this->a; point b=this->b; point c=this->c;
    result.emplace_back(this->a, this->b);
    result.emplace_back(this->b, this->c);
    result.emplace_back(this->a, this->c);

    return result;
}

template<class T>
bool triangle<T>::operator==(const triangle<T>& other) {
    //point a=this->a; point b=this->b; point c=this->c;
    return (this->a == other.a && this->b == other.b && this->c == other.c);
}

// explicit instantiation

template class point<double>;
template class point<qnnum::Qnnum>;

template class edge<double>;
template class edge<qnnum::Qnnum>;

template class circle<double>;
template class circle<qnnum::Qnnum>;

template class triangle<double>;
template class triangle<qnnum::Qnnum>;

//template<> bool circle<qnnum.Qnnum>::infinite<qnnum::Qnnum>() {
//    return radius == number::inf<qnnum::Qnnum>();
//}
