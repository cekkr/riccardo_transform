#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>
#include <limits>
#include <map>
#include <complex>

#include "riccardoTransform.cpp"

int main() {
    int length = 100;
    double refPi = PI / (length / 2); 
    std::vector<double> data(length);

    for (int x = 0; x < length; ++x) {
        data[x] = std::sin(refPi * x) + 
                  (std::sin((refPi * x * 2) + (PI / 4)) * 0.5) + 
                  std::sin(refPi * x * 3) + 
                  std::sin(refPi * x * 8);
    }

    std::vector<Sinusoid> sinusoids_1, sinusoids_2;
    std::vector<double> residue, resultant;

    std::tie(sinusoids_1, residue, resultant) = decompose_sinusoid(data, 2, 8, 10, 1);
    std::tie(sinusoids_2, residue, resultant) = decompose_sinusoid(residue, 2, 8, 10, 1);

    std::cout << std::endl << "Sinusoids 1:" << std::endl;
    for (const auto& s : sinusoids_1) {
        std::cout << "Frequency: " << s.frequency << ", Amplitude: " << s.amplitude << ", Phase: " << s.phase << std::endl;
    }

    std::cout << std::endl << "Sinusoids 2:" << std::endl;
    for (const auto& s : sinusoids_2) {
        std::cout << "Frequency: " << s.frequency << ", Amplitude: " << s.amplitude << ", Phase: " << s.phase << std::endl;
    }

    std::vector<Sinusoid> total_sinusoids = combine_sinusoids(sinusoids_1, sinusoids_2);

    std::cout << std::endl << "Total sinusoids:" << std::endl;
    for (const auto& s : total_sinusoids) {
        std::cout << "Frequency: " << s.frequency << ", Amplitude: " << s.amplitude << ", Phase: " << s.phase << std::endl;
    }

    return 0;
}