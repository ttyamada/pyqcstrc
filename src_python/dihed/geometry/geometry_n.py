# implementation of point<T>::point, edge<T>::edge, triangle<T>::triangle classes 
import crsys
import qnnum
import number
from number_hpp import Number as number

def point():
    a = None
    zero = number.num(0)
    x = zero
    y = zero #constructor

def finite():
    # point a=*this; a.x a.y
    #T inf = std::numeric_limits<T>::infinity()
    inf = number.inf()
    return abs(self.x) != inf and abs(self.y) != inf

def slope(a, b):
    a_ = None
    #return a.x - b.x == number::zero<T>() ? number::inf<T>() :
    #    (a.y - b.y) / (a.x - b.x)
    return number.inf() if a.x - b.x == number.num(0) else (a.y - b.y) / (a.x - b.x)
    #return a.x - b.x == number::num<T>(0) ? std::numeric_limits<T>::infinity() :
    #    (a.y - b.y) / (a.x - b.x)

def midpoint(a, b):
    a_ = None
    return point((a.x + b.x) / number.num(2), (a.y + b.y) / number.num(2))

def distance_squared(a):
    dx = x - a.x
    dy = y - a.y
    return (dx * dx + dy * dy)

def contains(p):
    dx = (p.x - center.x)
    dy = (p.y - center.y)
    dist = dx * dx + dy * dy

    return dist < radius

def infinite():
    return radius == number.inf()
    #return radius == std::numeric_limits<T>::infinity()

def triangle():
    a = point()
    b = point()
    c = point() # constructor

def valid():
    #point a = this->a; point b=this->b; point c=this->c
    if not self.a.finite() or not self.b.finite() or not self.c.finite():
        return False

    #     Compute the triangle validity, e.g., whether the points are collinear.
    #    ** We can do this by checking the area using the shoelace formula.
    #    ** The determinant of a matrix with 2 vectors is the area of the
    #    ** parallelogram spanning these vectors.
    #    **
    #    ** 1/2 of this area yields the area of the triangle spanning these vectors.
    #     
    # area is given by the determinant
    # | b.x - a.x  c.x - a.x |
    # | b.y - a.y  c.y - a.y |
    a_ = None
    area = ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / number.num(2)

    # Check the actual area, not signed area
    epsilon = number.eps()
    return abs(area) > epsilon #std::numeric_limits<T>::epsilon();

def has_vertex(p):
    # point a=this->a; point b=this->b; pointc=this->c
    return (self.a is p or self.b is p or self.c is p)

def circumcircle():
    #point a=this->a; point b=this->b; point c=this->c
    # Circumcenter can be found by finding the intersection of two
    # perpendicular bisectors

    # Check for collinearity   T zero = number::get_zerp(a)
    a_ = None
    zero = number.num(0) # number.
    one = number.num(1) # number.
    if not valid():
        #return circle(point<T>(zero, zero), std::numeric_limits<T>::infinity())
        return circle(point(zero, zero), number.inf())

    slope_ab = point.slope(self.a, self.b)
    slope_bc = point.slope(self.b, self.c)
    slope_ac = point.slope(self.a, self.c)

    midpoint_1 = point(self.a, self.b)

    slope_1 = slope_ab

    midpoint_2 = point(self.b, self.c)

    slope_2 = slope_bc

    if slope_1 is zero:
        midpoint_1 = point.midpoint(self.a, self.c)
        slope_1 = slope_ac
    elif slope_2 is zero:
        midpoint_2 = point.midpoint(self.a, self.c)
        slope_2 = slope_ac

    # Calculate the slopes of the perpendicular bisectors
    m1 = -one / slope_1
    m2 = -one / slope_2

    b1 = midpoint_1.y - m1 * midpoint_1.x
    b2 = midpoint_2.y - m2 * midpoint_2.x

    #    
    #    ** Bisectors intersect when y1 = y2,
    #    **   => m1(x) + b1 = m2(x) + b2
    #    **   => x(m1 - m2) = b2 - b1
    #    **   => x = (b2 - b1) / (m1 - m2)
    #     
    if (m1 - m2) == number.zero():
        print("m1 ", end = '')
        print(m1, end = '')
        print(" m2 ", end = '')
        print(m2, end = '')
        print(" m1-m2", end = '')
        print((m1 - m2), end = '')
        print()
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    center = point(x, y) # constructor

    #    
    #    ** While by definition of a circumcircle any point will be on the circle,
    #    ** due to floating-point precision issues there may be a small difference
    #    ** from a point to the circle's center.
    #    **
    #    ** Due to this, a point on the circumference may be detected as within the
    #    ** circle. Similar issues still persist when using other formulas for the
    #    ** radius (Ex: https://mathworld.wolfram.com/Circumradius.html).
    #    **
    #    ** Therefore, we're electing to find the smallest radius, such that no
    #    ** point on the circumference will be detected as within the circle.
    #     
    radius = min(min(a.distance_squared(center), b.distance_squared(center)), c.distance_squared(center)) # this works for qnnum.Qnnum?

    return circle(center, radius)

def has_edge(e):
    list = edges()
    for v in list:
        # Undirected edges, so the order does not matter
        if (v.a.equals_to(e.a) and v.b.equals_to(e.b)) or (v.a.equals_to(e.b) and v.b.equals_to(e.a)):
            return True

    return False

def edges():
    result = []
    #point a=this->a; point b=this->b; point c=this->c
    result.emplace_back(self.a, self.b)
    result.emplace_back(self.b, self.c)
    result.emplace_back(self.a, self.c)

    return list(result)
