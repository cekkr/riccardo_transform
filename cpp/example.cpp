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

    std::vector<double> curData = data;
    std::vector<Sinusoid> sinusoids;
    for(int x = 0; x < 10; x++) {
        std::vector<Sinusoid> _sinusoids;
        std::vector<double> _residue, _resultant;
        std::tie(_sinusoids, _residue, _resultant) = decompose_sinusoid(curData, 2, 8, 10, 1);

        if(false){
            std::cout << std::endl << "Cur sinusoids:" << std::endl;
            for (const auto& s : _sinusoids) {
                std::cout << "Frequency: " << s.frequency << ", Amplitude: " << s.amplitude << ", Phase: " << s.phase << std::endl;
            }
        }

        if(sinusoids.size() == 0)
            sinusoids = _sinusoids;
        else
            sinusoids = combine_sinusoids(sinusoids, _sinusoids);

        curData = _residue;
    }

    std::cout << std::endl << "Total sinusoids:" << std::endl;
    for (const auto& s : sinusoids) {
        std::cout << "Frequency: " << s.frequency << ", Amplitude: " << s.amplitude << ", Phase: " << s.phase << std::endl;
    }

    return 0;
}