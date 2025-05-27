# ====================================================================================================
# Produced by the Free Edition of C++ to Python Converter.
# Purchase a Premium Edition license at:
# https://www.tangiblesoftwaresolutions.com/order/order-cplus-to-python.html
# ====================================================================================================

#template<typname T>
#class delaunay {
class delaunay: #this class replaces the original namespace 'delaunay'
    @staticmethod
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
        inf = std::numeric_limits.infinity()
        return triangle(point(-inf, -inf), point(0.0, inf), point(inf, 0.0))

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
        inf = std::numeric_limits.infinity()

        finite_points = 1 if t.a.finite() + t.b.finite() + t.c.finite() else 0

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
                v1.copy_from(t.a);
                v2.copy_from(t.c)
            elif t.c.finite():
                f.copy_from(t.c)
                v1.copy_from(t.a)
                v2.copy_from(t.b)

            if v1.y == inf or v2.y == inf:
                if v1.x == inf or v2.x == inf:
                    # Vertices: { (0, inf), (inf, 0) }
                    # y = -x + b
                    b = f.y + f.x
                    if p.y + p.x > b:
                        return True
                else:
                    # Vertices: { (0, inf), (-inf, -inf) }
                    # y = 3x + b
                    b = f.y - 3.0 * f.x
                    if p.y - 3.0 * p.x > b:
                        return True
            else:
                # Vertices: { (-inf, -inf), (inf, 0) }
                # y = 1/3x + b
                b = f.y - f.x / 3.0
                if p.y - p.x / 3.0 < b:
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
            if f.y == inf:
                # Vertex: (0, inf)
                # The circle interior is always above the line
                if p.y - m * p.x > b:
                    return True
            elif f.x == inf:
                # Vertex: (inf, 0)
                if m >= 0.0:
                    if p.y - m * p.x < b:
                        return True
                elif p.y - m * p.x > b:
                    return True
            else:
                # Vertex: (-inf, -inf)
                if m >= 1.0:
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
        triangulation = []

        super = delaunay.super_triangle()
        triangulation.append(super)

        # Add each point to the triangulation
        for p in points:
            print(p.x, end = '')
            print(" ", end = '')
            print(p.y, end = '')
            print()
            bad_set = []

            # Find out which triangles are invalidated when adding this point
            for t in triangulation:
                circumcircle = t.circumcircle()
                if not circumcircle.infinite():
                    #std::cout<<"circumcircle.infinite() "<<circumcircle.infinite()<<'\n'
                    if circumcircle.contains(p):
                        bad_set.append(t)
                else:
                    #std::cout<<"circumcircle.infinite() "<<circumcircle.infinite()<<'\n'
                    if delaunay.halfplane_contains(t, p):
                        bad_set.append(t)

            polygon = []

            # Find edges not shared with any other flagged triangles
            for i, _ in enumerate(bad_set):
                edges = bad_set[i].edges()

                for e in edges:
                    shared_edge = False
                    for j, _ in enumerate(bad_set):
                        if i == j:
                            continue

# C++ TO PYTHON CONVERTER TASK: The following line was determined to contain a copy constructor call
# - this should be verified and a copy constructor should be created:
# ORIGINAL LINE: if(bad_set[j].has_edge(e))
                        if bad_set[j].has_edge(edge(e)):
                            shared_edge = True
                            break

                    if not shared_edge:
                        polygon.append(e)

            # Remove all bad triangles from the triangulation
            predicate = lambda t : t in bad_set

# C++ TO PYTHON CONVERTER TASK: There is no direct equivalent to the STL vector 'erase' method in Python:
            triangulation.erase(std::remove_if(triangulation.begin(), triangulation.end(), predicate), triangulation.end())

            # Connect edges to our point to form a new triangle
            for j, _ in enumerate(polygon):
                e = polygon[j]

                triangulation.emplace_back(e.a, e.b, p)

        # Remove any remaining triangle connected to the super triangle
        predicate = lambda t : t.has_vertex(super.a) or t.has_vertex(super.b) or t.has_vertex(super.c)

# C++ TO PYTHON CONVERTER TASK: There is no direct equivalent to the STL vector 'erase' method in Python:
        triangulation.erase(std::remove_if(triangulation.begin(), triangulation.end(), predicate), triangulation.end())

        return list(triangulation)
