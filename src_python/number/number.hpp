#pragma once
#include <iostream>
#include <limits>
#include <numeric>

#include "qnnum.hpp"

// class number includes conventional number and qnnumber
// interface
namespace number {
	template<class T> static T num(int a);
	template<class T> static T eps();
	template<class T> static T zero();
	template<class T> static T inf();
	template<class T> static T nan();
	template<class T> static bool is_inf(T a);
	template<class T> T abs(T a);
	template<class T> T min(T a, T b);
	template<class T> T max(T a, T b);
};

// specialization
template<> double number::num<double>(int a) {
	double c = (double)a; // float, double, long int etc.
	return c;
};
template<> qnnum::Qnnum number::num<qnnum::Qnnum>(int a){
	int d[] = {a,0,1};
	qnnum::Qnnum c = qnnum::Qnnum(d);  // Qnnum
	return c;
};

template<> double number::eps<double>() {
	return std::numeric_limits<double>::epsilon();
}

template<> qnnum::Qnnum number::eps<qnnum::Qnnum>() {
	return qnnum::zero();
}

template<> double number::zero<double>() {
	return 0.0;
}

template<> qnnum::Qnnum number::zero<qnnum::Qnnum>() {
	return qnnum::zero();
}

template<> double number::inf<double>() {
	return std::numeric_limits<double>::infinity();
}

template<> qnnum::Qnnum number::inf<qnnum::Qnnum>() {
	return qnnum::inf();
}

template<> double number::nan<double>() {
	return std::nan("");
}

template<> qnnum::Qnnum number::nan<qnnum::Qnnum>() {
	return qnnum::nan();
}

template<> bool number::is_inf<double>(double a) {
	return a==inf<double>();
}

template<> bool number::is_inf<qnnum::Qnnum>(qnnum::Qnnum a) {
	return a==qnnum::inf();
}

// spetialization for qnnum::Qnnum
template<> double number::abs<double>(double a) {
	return std::abs(a);
}

template<> qnnum::Qnnum number::abs<qnnum::Qnnum>(qnnum::Qnnum a) {
    return qnnum::abs(a);
}

template<> double number::min<double>(double a, double b) {
	return std::min(a,b);
}

template<> qnnum::Qnnum number::min<qnnum::Qnnum>(qnnum::Qnnum a, qnnum:: Qnnum b) {
	return qnnum::min(a,b);
}

template<> double number::max<double>(double a, double b) {
	return std::max(a,b);
}

template<> qnnum::Qnnum number::max<qnnum::Qnnum>(qnnum::Qnnum a, qnnum:: Qnnum b) {
	return qnnum::max(a,b);
}