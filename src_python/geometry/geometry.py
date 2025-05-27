# ====================================================================================================
# Produced by the Free Edition of C++ to Python Converter.
# Purchase a Premium Edition license at:
# https://www.tangiblesoftwaresolutions.com/order/order-cplus-to-python.html
# ====================================================================================================

def __init__():
    self.x = 0
    self.y = 0
def __init__(x, y):
    self.x = x
    self.y = y

# C++ TO PYTHON CONVERTER WARNING: 'const' methods are not available in Python:
# ORIGINAL LINE: bool point::finite() const
def finite():
    inf = std::numeric_limits.infinity()
    return abs(x) != inf and abs(y) != inf

def slope(a, b):
    return std if a.x - b.x == 0.0 else :numeric_limits.infinity() : (a.y - b.y) / (a.x - b.x)

def midpoint(a, b):
    return point((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)

# C++ TO PYTHON CONVERTER WARNING: 'const' methods are not available in Python:
# ORIGINAL LINE: double point::distance_squared(const point& a) const
def distance_squared(a):
    dx = x - a.x
    dy = y - a.y
    return (dx * dx + dy * dy)

def __init__(a, b):
    self.a = point(a)
    self.b = point(b)

def __init__(center, radius):
    self.center = point(center)
    self.radius = radius

# C++ TO PYTHON CONVERTER WARNING: 'const' methods are not available in Python:
# ORIGINAL LINE: bool circle::contains(const point& p) const
def contains(p):
    dx = (p.x - center.x)
    dy = (p.y - center.y)
    dist = dx * dx + dy * dy

    return dist < radius

# C++ TO PYTHON CONVERTER WARNING: 'const' methods are not available in Python:
# ORIGINAL LINE: bool circle::infinite() const
def infinite():
    return radius == std::numeric_limits.infinity()

def __init__(a, b, c):
    self.a = point(a)
    self.b = point(b)
    self.c = point(c)

# C++ TO PYTHON CONVERTER WARNING: 'const' methods are not available in Python:
# ORIGINAL LINE: bool triangle::valid() const
def valid():
    if not a.finite() or not b.finite() or not c.finite():
        return False

    #     Compute the triangle validity, e.g., whether the points are collinear.
    #    ** We can do this by checking the area using the shoelace formula.
    #    ** The determinant of a matrix with 2 vectors is the area of the
    #    ** parallelogram spanning these vectors.
    #    **
    #    ** 1/2 of this area yields the area of the triangle spanning these vectors.
    #     

    # | b.x - a.x  c.x - a.x |
    # | b.y - a.y  c.y - a.y |
    area = ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / 2.0

    # Check the actual area, not signed area
    return abs(area) > std::numeric_limits.epsilon()

# C++ TO PYTHON CONVERTER WARNING: 'const' methods are not available in Python:
# ORIGINAL LINE: bool triangle::has_vertex(const point& p) const
def has_vertex(p):
    return (a is p or b is p or c is p)

# C++ TO PYTHON CONVERTER WARNING: 'const' methods are not available in Python:
# ORIGINAL LINE: circle triangle::circumcircle() const
def circumcircle():
    # Circumcenter can be found by finding the intersection of two
    # perpendicular bisectors

    # Check for collinearity
    if not valid():
        return circle(point(0, 0), std::numeric_limits.infinity())

    slope_ab = point.slope(a, b)
    slope_bc = point.slope(b, c)
    slope_ac = point.slope(a, c)

    midpoint_1 = point.midpoint(a, b)

    slope_1 = slope_ab

    midpoint_2 = point.midpoint(b, c)

    slope_2 = slope_bc

    if slope_1 == 0.0:
# C++ TO PYTHON CONVERTER TASK: The following line was determined to be a copy assignment (rather than a reference assignment) - this should be verified and a 'copy_from' method should be created:
# ORIGINAL LINE: midpoint_1 = point::midpoint(a, c);
        midpoint_1.copy_from(point.midpoint(a, c))
        slope_1 = slope_ac
    elif slope_2 == 0.0:
# C++ TO PYTHON CONVERTER TASK: The following line was determined to be a copy assignment (rather than a reference assignment) - this should be verified and a 'copy_from' method should be created:
# ORIGINAL LINE: midpoint_2 = point::midpoint(a, c);
        midpoint_2.copy_from(point.midpoint(a, c))
        slope_2 = slope_ac

    # Calculate the slopes of the perpendicular bisectors
    m1 = -1 / slope_1
    m2 = -1 / slope_2

    b1 = midpoint_1.y - m1 * midpoint_1.x
    b2 = midpoint_2.y - m2 * midpoint_2.x

    #    
    #    ** Bisectors intersect when y1 = y2,
    #    **   => m1(x) + b1 = m2(x) + b2
    #    **   => x(m1 - m2) = b2 - b1
    #    **   => x = (b2 - b1) / (m1 - m2)
    #     
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    center = point(x, y)

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
    radius = min(min(a.distance_squared(center), b.distance_squared(center)), c.distance_squared(center))

    return circle(center, radius)

# C++ TO PYTHON CONVERTER WARNING: 'const' methods are not available in Python:
# ORIGINAL LINE: bool triangle::has_edge(edge e) const
def has_edge(e):
    list = edges()
    for v in list:
        # Undirected edges, so the order does not matter
        if (v.a.equals_to(e.a) and v.b.equals_to(e.b)) or (v.a.equals_to(e.b) and v.b.equals_to(e.a)):
            return True

    return False

# C++ TO PYTHON CONVERTER WARNING: 'const' methods are not available in Python:
# ORIGINAL LINE: list<edge> triangle::edges() const
def edges():
    result = []
    result.emplace_back(a, b)
    result.emplace_back(b, c)
    result.emplace_back(a, c)

    return list(result)