#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <limits>
#include <map>
#include <complex>

namespace {
    // Constants
    const double PI = std::acos(-1);
}

class Derivator {
 public:
  Derivator(int n) : n_(n), derivatives_(n + 1) {
    for(int i = 0; i < n; i++){
        numbers_.push_back(0);
    }    
  }

  // Adds a number to the series and calculates the derivatives
  void addNumber(double number) {
    // Shift previous values to higher order derivatives    
    for (int i = n_; i > 0; --i) {       
        derivatives_[i] = derivatives_[i - 1];
    }     
    derivatives_[0] = number;
    
    // Calculate the new derivatives
    for (int i = 1; i <= n_; ++i) {
        derivatives_[i] = derivatives_[i - 1] - numbers_.back();
    }

    numbers_.push_back(number);
  }

  // Returns the n derivatives calculated so far
  std::vector<double> getDerivatives() const {
    return std::vector<double>(derivatives_.begin() + 1, derivatives_.end());
  }

 private:
  int n_;                       // Number of derivatives to calculate
  std::vector<double> numbers_;  // Numbers inserted in the series
  std::vector<double> derivatives_; // Calculated derivatives
};

double calculate_mean(const std::vector<double>& numbers) {
    double sum = std::accumulate(numbers.begin(), numbers.end(), 0.0);
    return std::abs(sum / numbers.size());
}

double calculate_mean_abs(const std::vector<double>& numbers) {
    double sum = 0;

    for (double number : numbers) {
        sum += std::abs(number);
    }

    return sum / numbers.size();
}

double calculateTrapezoidArea(double base, double height1, double height2, bool absolute=false) {
    double area = 0.0;

    // If both heights are positive or both are negative
    if ((height1 >= 0 && height2 >= 0) || (height1 <= 0 && height2 <= 0)) {
        // Calculate the area of the trapezoid normally
        area = (base * (height1 + height2)) / 2.0;

        // If both heights are negative, the area is negative
        if (height1 < 0 && height2 < 0) {
            area = -std::abs(area);
        }
    } else {
        // If one height is positive and the other is negative
        double positiveHeight = height1 >= 0 ? height1 : height2;
        double negativeHeight = height1 < 0 ? height1 : height2;

        // Calculate the area of the triangle above the base (positive)
        double positiveTriangleArea = (base * positiveHeight) / 2.0;

        // Calculate the area of the triangle below the base (negative)
        double negativeTriangleArea = (base * std::abs(negativeHeight)) / 2.0;

        // Sum positive and negative area
        if(absolute)
            area = positiveTriangleArea + negativeTriangleArea;
        else
            area = positiveTriangleArea - negativeTriangleArea;
    }

    if(absolute)
        area = fabs(area);

    return area;
}


double calculate_max_peak(const std::vector<double>& numbers, const std::vector<double>& original, const std::vector<std::size_t> peaks){
    double totArea = 0;
    double prevDiff = 0;
    double prevPeak = -1;
    int i = 0;
    for (int peak : peaks) {
        double num = numbers[i++];
        double orig = original[peak];
        
        double diff = num; 

        if(orig < 0) diff = -diff;
        if((diff < 0 && orig > 0) || (diff > 0 && orig < 0)) diff *= -1;

        if(prevPeak >= 0){
            double area = calculateTrapezoidArea((double)(peak-prevPeak), prevDiff, diff, false);              
            totArea += area;
        }

        prevDiff = diff;
        prevPeak = peak;
    }

    totArea /= numbers.size();     
    return totArea;
}

// Helper function to find peaks using Boost
std::vector<std::size_t> find_peaks(const std::vector<double>& array, double threshold=0.5) {
    std::vector<std::size_t> peaks;    
    if (array.size() < 2) { 
        return peaks; // Not enough data to find peaks
    }

    enum class Trend {
        Increasing,
        Decreasing,
        None
    };
   
    Derivator derivator(2); 
    int referenceDerivation = 0;
    Trend currentTrend = Trend::None;
    for (std::size_t i = 0; i < array.size(); ++i) {        

        derivator.addNumber(array[i]);         
               
        std::vector<double> derivatives = derivator.getDerivatives();

        if(false){
            std::cout << "Derivatives: ";
            for (double derivative : derivatives) {
                std::cout << derivative << " ";
            }
            std::cout << std::endl; 
        }

        //std::cout << absDiff << " > " << avgAbsAvgDiff*threshold << " \t(" << avgDiff << ")" << std::endl;
        if (derivatives[referenceDerivation] > 0) {
            if (currentTrend == Trend::Decreasing) {
                peaks.push_back(i); // Peak found
            } 
            currentTrend = Trend::Increasing;
        } else if (derivatives[referenceDerivation] < 0) {
            if (currentTrend == Trend::Increasing) {
                peaks.push_back(i); // Peak found
            } 
            currentTrend = Trend::Decreasing;
        } else { 
            currentTrend = Trend::None;
            //peaks.push_back(i);
        }                      
    }

    //std::cout << "Number of peaks: " << peaks.size() << std::endl;
    return peaks;
}

std::vector<double> generate_sinusoid(double freq, double amplitude, double phase, int length) {
    std::vector<double> sinusoid(length);
    for (int x = 0; x < length; ++x) {
        sinusoid[x] = amplitude * std::sin((x * freq) + phase);
    }
    return sinusoid;
}

double calculate_sin(double freq, double amplitude, double phase, double x) {
    return amplitude * std::sin((x * freq) + phase);
}

double find_best_phase(const std::vector<double>& residue, double frequency, double amplitude, int precision, const std::vector<std::size_t>& peak_indices) {
    std::vector<double> phases = {0, PI, PI * 1.75};
    double best_phase = -1;
    int num_identical = 0;

    for (int i = 0; i < precision; ++i) {
        std::vector<double> errors;
        for (double phase : phases) {
            std::vector<double> diffs;
            for (std::size_t peak : peak_indices) {
                double val = calculate_sin(frequency, amplitude, phase, peak);
                double res = residue[peak];
                double diff = res-val;
                diffs.push_back(val);
            }

            double mean = calculate_max_peak(diffs, residue, peak_indices);
            //double mean = calculate_mean_abs(diffs);
            std::cout << "!(" << frequency << ")" << " \tAmplitude: " << amplitude  << " \tPhase: " << phase << " \tArea: " << mean << std::endl;
            errors.push_back(mean); // *std::max_element(diffs.begin(), diffs.end())
        }

        auto it = std::max_element(errors.begin(), errors.end());
        double best = phases[std::distance(errors.begin(), it)];

        if (best == best_phase) {
            num_identical++;
            if (num_identical >= precision / 2) {
                break;
            }
        }

        best_phase = best;

        double step = (*std::max_element(phases.begin(), phases.end()) - *std::min_element(phases.begin(), phases.end())) / 2;
        phases = {best_phase - step / 2, best_phase, best_phase + step / 2};
        phases.erase(std::remove_if(phases.begin(), phases.end(), [](double phase){ return phase < 0 || phase > 2 * PI; }), phases.end());

        if (phases.size() == 1) {
            break;
        }
    }

    return best_phase;
}

double find_best_amplitude(const std::vector<double>& residue, double frequency, double phase, int precision, const std::vector<std::size_t>& peak_indices, double baseAmplitude=1.0) {
    if(baseAmplitude == 0) 
        baseAmplitude = 1.0;
    
    std::vector<double> amplitudes = {0, baseAmplitude, baseAmplitude*2};
    double best_amplitude = -1;
    int num_identical = 0;

    for (int i = 0; i < precision; ++i) {
        std::vector<double> errors;
        for (double amplitude : amplitudes) {
            std::vector<double> diffs;
            for (std::size_t peak : peak_indices) {
                double val = calculate_sin(frequency, amplitude, phase, peak);
                double res = residue[peak];
                double diff = res-val;
                diffs.push_back(diff);
            }

            double mean = calculate_max_peak(diffs, residue, peak_indices);
            //double mean = calculate_mean_abs(diffs);
            std::cout << "(" << frequency << ")" << " \tAmplitude: " << amplitude  << " \tPhase: " << phase << " \tArea: " << mean << std::endl;
            errors.push_back(mean); // *std::max_element(diffs.begin(), diffs.end())
        }

        auto it = std::max_element(errors.begin(), errors.end());
        double best = amplitudes[std::distance(errors.begin(), it)];

        if (best_amplitude == best) {
            num_identical++;
            if (num_identical >= precision / 2) {
                break;
            }
        }

        best_amplitude = best;

        double step = (*std::max_element(amplitudes.begin(), amplitudes.end()) - *std::min_element(amplitudes.begin(), amplitudes.end())) / 2;
        amplitudes = {best_amplitude - step / 2, best_amplitude, best_amplitude + step / 2};
        amplitudes.erase(std::remove_if(amplitudes.begin(), amplitudes.end(), [](double amp){ return amp < 0 || amp > 2; }), amplitudes.end());

        if (amplitudes.size() == 1) {
            break;
        }
    }

    return best_amplitude;
}

std::vector<std::size_t> union_without_duplicates(const std::vector<std::size_t>& list1, const std::vector<std::size_t>& list2) {
    std::vector<std::size_t> combined_list = list1;
    for (std::size_t item : list2) {
        if (std::find(combined_list.begin(), combined_list.end(), item) == combined_list.end()) {
            combined_list.push_back(item);
        }
    }
    return combined_list;
}

struct Sinusoid {
    double frequency;
    double phase;
    double amplitude;
};

std::tuple<std::vector<Sinusoid>, std::vector<double>, std::vector<double>> 
decompose_sinusoid(const std::vector<double>& data, double halving = 2.0, int precision = 6, 
                   int max_halvings = 10, double reference_size = 1, double negligible = 0.01) {
    int length = data.size();
    std::vector<Sinusoid> sinusoids;
    std::vector<double> residue(data); 
    std::vector<double> resultant(length, 0.0); 

    double relative_frequency = (2 * PI) / length;
    reference_size = reference_size <= 1 ? reference_size : std::pow(2, reference_size);
    double frequency = reference_size * relative_frequency;

    auto generate_sin = [&](double frequency, const std::vector<double>& data) {
        double amplitude = 1;
        double phase = 0;

        std::vector<double> base_wave = generate_sinusoid(frequency, amplitude, phase, length);
        std::vector<double> wave(length);
        for (int i = 0; i < length; ++i) {
            wave[i] = base_wave[i] - data[i];
        }
        std::vector<std::size_t> peaks = find_peaks(wave);

        base_wave = generate_sinusoid(frequency, amplitude, PI, length);
        for (int i = 0; i < length; ++i) {
            wave[i] = base_wave[i] - data[i];
        }
        std::vector<std::size_t> peaks_2 = find_peaks(wave);

        peaks = union_without_duplicates(peaks, peaks_2);

        double _phase = -1;
        double _amplitude = -1;
        for (int i = 0; i < precision / 2; ++i) {
            phase = find_best_phase(residue, frequency, amplitude, precision, peaks);
            amplitude = find_best_amplitude(residue, frequency, phase, precision, peaks, amplitude);

            if ((std::abs(_phase - phase) + std::abs(_amplitude - amplitude)) / 2 < negligible) {
                break;
            }

            _phase = phase;
            _amplitude = amplitude;
        }

        std::vector<double> sinusoid = generate_sinusoid(frequency, amplitude, phase, length);
        std::vector<double> diff(length);
        for (int i = 0; i < length; ++i) {
            diff[i] = data[i] - sinusoid[i];
        }

        //double mean_diff = calculate_mean(diff);
        double mean_diff = calculate_max_peak(diff, data, peaks);

        return std::make_tuple(amplitude, phase, mean_diff, diff, sinusoid);
    };

    double mean_residue = 0; 
    double prev_frequency = frequency;
    double next_frequency = frequency * 2;

    for (int i = 0; i < max_halvings; ++i) {
        mean_residue = calculate_mean_abs(residue);
        if (mean_residue < negligible) {
            break; 
        }

        double amplitude, phase, mean_diff;
        std::vector<double> diff, sinusoid;

        std::tie(amplitude, phase, mean_diff, diff, sinusoid) = generate_sin(frequency, residue);

        bool no_frequency_halving = false;
        double min_freq = prev_frequency;
        if (prev_frequency != frequency) {
            double freq = frequency;
            std::vector<double> best_sin; 
            std::vector<double> _diff, _sinusoid;
            double reference_mean = mean_residue;            

            for (int j = 0; j < precision / 2; ++j) {
                freq = (freq + min_freq) / 2;
                double _amplitude, _phase, _mean_diff;                
                std::tie(_amplitude, _phase, _mean_diff, _diff, _sinusoid) = generate_sin(freq, residue);

                if (_mean_diff < mean_diff || _mean_diff < reference_mean) {
                    break; 
                }

                if (mean_residue - _mean_diff > mean_diff - _mean_diff) {
                    min_freq = frequency;
                } else {
                    min_freq = frequency / halving;
                }

                reference_mean = _mean_diff;
                best_sin = {freq, _amplitude, _phase, _mean_diff}; 
            }

            if (!best_sin.empty()) {
                frequency = best_sin[0];
                amplitude = best_sin[1];
                phase = best_sin[2];
                mean_diff = best_sin[3]; 
                diff.swap(_diff);
                sinusoid.swap(_sinusoid);
                no_frequency_halving = true;
            }
        }

        double new_mean_residue = calculate_mean_abs(diff);
        if (new_mean_residue < mean_residue/2) {
            amplitude = 0; 
        } else {
            // Update resultant and residue
            for (int i = 0; i < length; ++i) {
                resultant[i] += sinusoid[i];
                residue[i] = diff[i];
            }
            mean_residue = mean_diff;
        }

        if (amplitude > 0) {
            sinusoids.push_back({frequency / relative_frequency, phase, amplitude});
        }

        prev_frequency = frequency;
        if (no_frequency_halving) {
            frequency = (frequency+min_freq)/2;
        } else {
            frequency = next_frequency;
            next_frequency *= halving;
        }
    }

    return std::make_tuple(sinusoids, residue, resultant);
}

std::vector<Sinusoid> combine_sinusoids(const std::vector<Sinusoid>& sinusoids1, 
                                        const std::vector<Sinusoid>& sinusoids2, 
                                        double minimum = 0.05) {
    std::map<double, std::complex<double>> combined_sinusoids;

    for (const auto& sinusoids : {sinusoids1, sinusoids2}) {
        for (const auto& sinusoid : sinusoids) {
            double freq = sinusoid.frequency;
            double amp = sinusoid.amplitude;
            double phase = sinusoid.phase;

            if (combined_sinusoids.count(freq)) {
                double existing_amp = std::abs(combined_sinusoids[freq]);
                double existing_phase = std::arg(combined_sinusoids[freq]);

                if (std::min(existing_amp, amp) / std::max(existing_amp, amp) < (minimum * 2)) {
                    if (amp > existing_amp) {
                        combined_sinusoids[freq] = std::polar(amp, phase); 
                    }
                } else {
                    combined_sinusoids[freq] += std::polar(amp, phase); 
                }
            } else {
                combined_sinusoids[freq] = std::polar(amp, phase); 
            }
        }
    }

    std::vector<Sinusoid> result;
    for (const auto& [freq, complex_sinusoid] : combined_sinusoids) {
        double amplitude = std::abs(complex_sinusoid);
        double phase = std::arg(complex_sinusoid);

        if (amplitude > minimum) {
            result.push_back({freq, phase, amplitude});
        }
    }

    std::sort(result.begin(), result.end(), 
              [](const Sinusoid& a, const Sinusoid& b) { return a.frequency < b.frequency; });

    return result;
}