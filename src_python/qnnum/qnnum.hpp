#pragma once
#include <array>
#include <numeric>
#include <cmath>
#include <iostream>
#include "crsys.hpp"

int N;

//namespace qnnum {

class Qnnum { 

    public:
    //std::array<int, 3> n;
    int n[3];

    //Qnnum(std::array<int, 3>& n_) {
    Qnnum(int n_[]) {
        for (int i=0; i<3; ++i) {
            n[i] = n_[i];
        }
    }

    Qnnum operator+(Qnnum& b) {
        int c1 = n[0] * b.n[2] + b.n[0] * n[2];
        int c2 = n[1] * b.n[2] + b.n[1] * n[2];
        int c3 = n[2] * b.n[2];
        int g = (int)std::gcd(c1, (int)std::gcd(c2, c3));
        int d1[] = {c1 / g, c2 / g, c3 / g};
        return Qnnum(d1);
    }

    Qnnum operator-(Qnnum& b) {
        int c1 = n[0] * b.n[2] - b.n[0] * n[2];
        int c2 = n[1] * b.n[2] - b.n[1] * n[2];
        int c3 = n[2] * b.n[2];
        int g = (int)std::gcd(c1, (int)std::gcd(c2, c3));
        int d1[] = {c1 / g, c2 / g, c3 / g};
        return *this = Qnnum(d1);
    }

    Qnnum operator+=(Qnnum& b) {
        return *this = *this + b;
    }

    Qnnum operator-=(Qnnum& b) {
        return *this = *this - b;
    }

    Qnnum operator*(Qnnum& b) {
        int c1 = n[0] * b.n[0] + n[1] * b.n[1] * N;
        int c2 = n[0] * b.n[1] + n[1] * b.n[0];
        int c3 = n[2] * b.n[2];
        int g = (int)std::gcd(c1, (int)std::gcd(c2, c3));
        int d1[] = {c1 / g, c2 / g, c3 / g};
        return *this = Qnnum(d1);
    }

    Qnnum operator*(const int b) {
        int c1 = n[0] * b;
        int c2 = n[1] * b;
        int c3 = n[2];
        int g = (int)std::gcd(c1, (int)std::gcd(c2, c3));
        int d1[] = {c1 / g, c2 / g, c3 / g};
        return *this = Qnnum(d1);
    }

    Qnnum operator/(Qnnum& b) {
        int c1 = b.n[0] * b.n[2];
        int c2 = -b.n[1] * b.n[2];
        int c3 = b.n[0] * b.n[0] - b.n[1] * b.n[1] * N;
        if (c3 == 0) throw std::runtime_error("ERROR: division by zero");
        int d1[] = {c1, c2, c3};
        return *this = *this * Qnnum(d1);
    }

    Qnnum operator/(int b) {
        int d1[] = {n[0], n[1], n[2] * b};
        return *this = Qnnum(d1);
    }

    bool operator==(Qnnum& b) {
        Qnnum c = *this - b;
        return (c.n[0] == 0 && c.n[1] == 0);
    }

    bool operator<(Qnnum& b) {
        Qnnum c = *this - b;
        return (std::signbit(c.n[0]) * c.n[0] * c.n[0] + std::signbit(c.n[1]) * c.n[1] * c.n[1] * N < 0);
    }

    bool operator>(Qnnum& b) {
        Qnnum c = *this - b;
        return (std::signbit(c.n[0]) * c.n[0] * c.n[0] + std::signbit(c.n[1]) * c.n[1] * c.n[1] * N > 0);
    }

    Qnnum operator~() {
        int d1[] = {-n[0], -n[1], n[2]};
        return *this = Qnnum(d1);
    }
};

namespace qnnum {

// Function implementations
void init() {
    N=crsys::N;
}

Qnnum zero() {
    int d1[] = {0, 0, 1};
    return Qnnum(d1);
}

Qnnum one() {
    int d1[] = {1,0,1};
    return Qnnum(d1);
}

Qnnum any(int  n_[]) {
    int d1[] = {n_[0],n_[1],n_[2]};
    return Qnnum(d1);
}

// printqnn
void printqnn(const std::string& str, Qnnum& a) {
    std::cout << str << " [" << a.n[0] << ", " << a.n[1] << ", " << a.n[2] << "]" << std::endl;
}

void printqnns(const std::string& str, std::vector<Qnnum>& a) {
    for (size_t i = 0; i < a.size(); ++i) {
        std::cout << str << " [" << a[i].n[0] << ", " << a[i].n[1] << ", " << a[i].n[2] << "]" << std::endl;
    }
}

};




