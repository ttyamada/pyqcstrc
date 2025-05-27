from geometry_n_hpp import Point, Edge, Triangle, Circle
from number_hpp import Number as number

# implementation
class delaunay: #this class replaces the original namespace 'delaunay'

    @staticmethod
# C++ TO PYTHON CONVERTER TASK: Python does not allow method overloads:
    def super_triangle():
        #         While we could arbitrarily form a super triangle that encompasses all points,
        #        ** if three points approach collinearity, their circumcircle will possibly
        #        ** extend beyond the super triangle if not big enough.
        #        **
        #        ** For reference, see post:
        #        ** https://math.stackexchange.com/questions/4001660
        #        **
        #        ** Big shout out to Hagen von Eitzen for suggesting symbolic vertices
        #        **
        #        ** So instead, we can just handle the special case where an infinite circle
        #        ** locally creates a half-plane
        #        
        #auto inf = std::numeric_limits<T>::infinity(); // only for conventional number
        a = None
        inf = number.inf() # geometry::point<T>
        zero = number.num(0) # geometry::point<T>
        #std::cout<<"inf "<< inf<<" zero "<<zero<<std::endl;  // for test
        #return triangle(point(-Globals.inf, -Globals.inf), point(Globals.zero, Globals.inf), point(Globals.inf, Globals.zero))
        return triangle(point(-inf, -inf), point(zero, inf), point(inf, zero))

    @staticmethod
    def wt_triangle(str, t):
        print(str,t.a.x," ",t.a.y," ",t.b.x," ",t.b.y," ",t.c.x," ",t.c.y)

    @staticmethod
    def halfplane_contains(t, p):
        #         If t circumscribes a circle with infinite radius, this circle is
        #        ** a line locally. We just need to find out which side of the half-plane
        #        ** a point is in order to tell whether it is interior to the circle or not.
        #        **
        #        ** 3 cases to handle:
        #        ** 3) all three vertices have infinite coordinates:
        #        **    then our circle simply contains all points
        #        **
        #        ** 2) 2 vertices have infinite coordinates,
        #        **
        #        ** 1) 1 vertex is infinite
        #        

        finite_points = 0
        #T inf = std::numeric_limits<T>::infinity();  // only for convensional number
        a = None
        inf = number.inf()
        finite_points = 1 if t.a.finite() + t.b.finite() + t.c.finite() else 0 # 0, 1 or 2

        # Case 3
        if finite_points == 0:
            return True

        if finite_points == 1:
            # Case 2: there are 2 infinite vertices
            f = point()
            v1 = point()
            v2 = point()
            if t.a.finite():
                f.copy_from(t.a)
                v1.copy_from(t.b)
                v2.copy_from(t.c)
            elif t.b.finite():
                f.copy_from(t.b)
                v1.copy_from(t.a)
                v2.copy_from(t.c)
            elif t.c.finite():
                f.copy_from(t.c)
                v1.copy_from(t.a)
                v2.copy_from(t.b)

            #if v1.y is Globals.inf or v2.y is Globals.inf:
            #    if v1.x is Globals.inf or v2.x is Globals.inf:
            if v1.y is inf or v2.y is inf:
                if v1.x is inf or v2.x is inf:
                    # Vertices: { (0, inf), (inf, 0) }
                    # y = -x + b
                    b = f.y + f.x
                    if p.y + p.x > b:
                        return True
                else:
                    # Vertices: { (0, inf), (-inf, -inf) }
                    # y = 3x + b
                    b = f.y - f.x * 3
                    if p.y - p.x * 3 > b:
                        return True
            else:
                # Vertices: { (-inf, -inf), (inf, 0) }
                # y = 1/3x + b
                b = f.y - f.x / 3
                if p.y - p.x / 3 < b:
                    return True
        elif finite_points == 2:
            f = point()
            v1 = point()
            v2 = point()
            if not t.a.finite():
                f.copy_from(t.a)
                v1.copy_from(t.b)
                v2.copy_from(t.c)
            elif not t.b.finite():
                f.copy_from(t.b)
                v1.copy_from(t.a)
                v2.copy_from(t.c)
            elif not t.c.finite():
                f.copy_from(t.c)
                v1.copy_from(t.b)
                v2.copy_from(t.a)

            # The line from v1 to v2 will be tangent to the circle
            m = point.slope(v1, v2)
            b = v1.y - m * v1.x
            #if f.y is Globals.inf:
            if f.y is inf:
                # Vertex: (0, inf)
                # The circle interior is always above the line
                if p.y - m * p.x > b:
                    return True
            #elif f.x is Globals.inf:
            elif f.x is inf:
                # Vertex: (inf, 0)
                if m >= 0:
                    if p.y - m * p.x < b:
                        return True
                elif p.y - m * p.x > b:
                    return True
            else:
                # Vertex: (-inf, -inf)
                if m >= 1:
                    if p.y - m * p.x > b:
                        return True
                elif p.y - m * p.x < b:
                    return True

        return False

    @staticmethod
    def triangulate(points):
        #        
        #        ** Bowyer-Watson algorithm
        #        ** Reference: https://en.wikipedia.org/wiki/Bowyer-Watson_algorithm
        #         
        triangles = []

        super = delaunay.super_triangle()
        wt_triangle("super ",super) # for test
        triangles.append(super)

        # Add each point to the triangles
        for p in points:
            print("p ",p.x," ",p.y)
            bad_set = []

            # Find out which triangles are invalidated when adding this point
            for t in triangles:
                circumcircle = t.circumcircle()
                print("circumcircle.radius ", end = '')
                print(circumcircle.radius, end = '')
                print()
                if not circumcircle.infinite():
                    wt_triangle("t ",t) # for test
                    if circumcircle.contains(p):
                        bad_set.append(t)
                else:
                    if delaunay.halfplane_contains(t, p):
                        bad_set.append(t)
            print("bad_set.size() ",len(bad_set))
            polygon = [] # polygon edge (double or Qnnum)

            # Find edges not shared with any other flagged triangles
            for i, _ in enumerate(bad_set):
                edges = bad_set[i].edges()

                for e in edges:
                    shared_edge = False
                    for j, _ in enumerate(bad_set):
                        if i == j:
                            continue
                        if bad_set[j].has_edge(edge(e)):
                            shared_edge = True
                            break

                    if not shared_edge:
                        polygon.append(e) # non-shared polygon edges

            # Remove all bad triangles from the triangles
            # definition of predicate only for float or double? ******** find -> find<T>
            predicate = lambda t : t in bad_set; #return true if t is in bad_set else false
            #triangles.erase(std::remove_if(triangles.begin(),triangles.end(),predicate), triangles.end())
            triangles=remove(triangles,predicate) # **** not implement yet

            # Connect edges to our point to form a new triangle
            for j, _ in enumerate(polygon):
                e = polygon[j]
                triangles.emplace_back(e.a, e.b, p)

        # Remove any remaining triangle connected to the super triangle
        predicate = lambda t : t.has_vertex(super.a) or t.has_vertex(super.b) or t.has_vertex(super.c)
        #triangles.erase(std::remove_if(triangles.begin(), triangles.end(), predicate), triangles.end())
        triangles=remove(triangles,predicate) # **** not implement yet

        return list(triangles)

    # explicit instantiation
    #    super_triangle() // explicit instantiation
    #    halfplane_contains(t, p)
    #    triangulate(points)
    #    super_triangle()
    #    halfplane_contains(t, p)
    #    triangulate(points)

    # for specialization use
    # template<> template triangle<double> super_triangle<double>() {} etc.


