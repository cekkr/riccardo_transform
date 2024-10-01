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

    Trend currentTrend = Trend::None;
    double avgDiff = 0;

    for (std::size_t i = 1; i < array.size(); ++i) {
        double diff = array[i] - array[i - 1];
        double absDiff = abs(diff);

        if(absDiff > avgDiff*threshold){
            if (diff > 0) {
                if (currentTrend == Trend::Decreasing) {
                    peaks.push_back(i - 1); // Peak found
                } 
                currentTrend = Trend::Increasing;
            } else if (diff < 0) {
                if (currentTrend == Trend::Increasing) {
                    peaks.push_back(i - 1); // Peak found
                } 

                currentTrend = Trend::Decreasing;
            } else { 
                //peaks.push_back(i - 1); // Peak found (ignore)
                currentTrend = Trend::None;
            }

            avgDiff = (absDiff + avgDiff) / 2;
        }
    }

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
                diffs.push_back(std::abs(val - res));
            }
            errors.push_back(*std::max_element(diffs.begin(), diffs.end()));
        }

        auto it = std::min_element(errors.begin(), errors.end());
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

double find_best_amplitude(const std::vector<double>& residue, double frequency, double phase, int precision, const std::vector<std::size_t>& peak_indices) {
    std::vector<double> amplitudes = {0, 1, 2};
    double best_amplitude = -1;
    int num_identical = 0;

    for (int i = 0; i < precision; ++i) {
        std::vector<double> errors;
        for (double amplitude : amplitudes) {
            std::vector<double> diffs;
            for (std::size_t peak : peak_indices) {
                double val = calculate_sin(frequency, amplitude, phase, peak);
                double res = residue[peak];
                diffs.push_back(std::abs(val - res));
            }
            errors.push_back(*std::max_element(diffs.begin(), diffs.end()));
        }

        auto it = std::min_element(errors.begin(), errors.end());
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
            phase = find_best_phase(data, frequency, amplitude, precision, peaks);
            amplitude = find_best_amplitude(data, frequency, phase, precision, peaks);

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
        double mean_diff = calculate_mean(diff);

        return std::make_tuple(amplitude, phase, mean_diff, diff, sinusoid);
    };

    double mean_residue = 1994; 
    double prev_frequency = frequency;
    double next_frequency = frequency * 2;

    for (int i = 0; i < max_halvings; ++i) {
        if (calculate_mean_abs(residue) < negligible) {
            break; 
        }

        double amplitude, phase, mean_diff;
        std::vector<double> diff, sinusoid;

        std::tie(amplitude, phase, mean_diff, diff, sinusoid) = generate_sin(frequency, residue);

        bool no_frequency_halving = false;
        if (prev_frequency != frequency) {
            double freq = frequency;
            std::vector<double> best_sin; 
            double reference_mean = mean_residue;
            double min_freq = prev_frequency;

            for (int j = 0; j < precision / 2; ++j) {
                freq = (freq + min_freq) / 2;
                double _amplitude, _phase, _mean_diff;
                std::vector<double> _diff, _sinusoid;
                std::tie(_amplitude, _phase, _mean_diff, _diff, _sinusoid) = generate_sin(freq, residue);

                if (_mean_diff > mean_diff || _mean_diff > reference_mean) {
                    break; 
                }

                if (std::abs(mean_residue - _mean_diff) > std::abs(mean_diff - _mean_diff)) {
                    min_freq = frequency;
                } else {
                    min_freq = frequency / halving;
                }

                reference_mean = _mean_diff;
                best_sin = {freq, _amplitude, _phase, _mean_diff}; 
                _diff.swap(diff); // Efficiently swap the vectors
                _sinusoid.swap(sinusoid);
            }

            if (!best_sin.empty()) {
                frequency = best_sin[0];
                amplitude = best_sin[1];
                phase = best_sin[2];
                mean_diff = best_sin[3]; 
                no_frequency_halving = true;
            }
        }

        if (mean_diff > mean_residue * 2) {
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
            frequency = prev_frequency * halving;
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