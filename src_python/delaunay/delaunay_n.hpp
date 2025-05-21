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

