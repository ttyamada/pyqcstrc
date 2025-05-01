#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <numeric>
#include "qnnum.hpp"

// Main function for testing or demonstration
int main() {
    Qnnum a = one();
    Qnnum b = zero();
    auto c = a + b;
    
    std::cout << "[" << c.n[0] << ", " << c.n[1] << ", " << c.n[2] << "]\n";
    return 0;
}
