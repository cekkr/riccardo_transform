#include <metal_stdlib>
using namespace metal;

struct Sinusoid {
    float frequency;
    float amplitude;
    float phase;
};

// Helper function to find peaks (using threadgroup memory)
uint find_peaks_kernel(device float* residue,
                        device int* peaks,
                        uint id [[thread_position_in_grid]],
                        uint size [[threads_per_grid]],
                        uint peaksCount) {

    if (id >= size || size < 2) {
        return 0;
    }

    enum Trend {
        Increasing,
        Decreasing,
        None
    };

    threadgroup_barrier(mem_flags::mem_device);

    Trend currentTrend = Trend::None;
    uint peakCount = 0;
    for (uint i = id + 1; i < size; i += size) {
        if (residue[i] > residue[i - 1]) {
            if (currentTrend == Trend::Decreasing) {
                peaks[peakCount++] = i - 1;
            }
            currentTrend = Trend::Increasing;
        } else if (residue[i] < residue[i - 1]) {
            if (currentTrend == Trend::Increasing) {
                peaks[peakCount++] = i - 1;
            }
            currentTrend = Trend::Decreasing;
        } else {
            currentTrend = Trend::None;
        }
        
        if (peakCount >= peaksCount) break;
    }

    if(peakCount < peaksCount) peaks[peakCount] = -1; // Mark the end of peaks
    return peakCount;
}

// Function to generate sinusoid
float generate_sinusoid(float freq, float amplitude, float phase, uint x) {
    return amplitude * sin((x * freq) + phase);
}

// Function to calculate sin value
float calculate_sin(float freq, float amplitude, float phase, float x) {
    return amplitude * sin((x * freq) + phase);
}

// Function to find the best phase
float find_best_phase(const device float* residue, float frequency, float amplitude, uint precision, device int* peak_indices, uint peakCount) {
    float PI = acos(-1.0f);

    float phases[3] = {0.0f, PI, PI * 1.75f};
    float best_phase = -1.0f;
    uint num_identical = 0;

    for (uint i = 0; i < precision; ++i) {
        float errors[3] = {0.0f, 0.0f, 0.0f};
        for (uint j = 0; j < 3; ++j) {
            float max_diff = 0.0f;
            for (uint k = 0; k < peakCount; ++k) {
                float val = calculate_sin(frequency, amplitude, phases[j], float(peak_indices[k]));
                float res = residue[peak_indices[k]];
                max_diff = fmax(max_diff, fabs(val - res));
            }
            errors[j] = max_diff;
        }

        uint min_index = 0;
        for (uint j = 1; j < 3; ++j) {
            if (errors[j] < errors[min_index]) {
                min_index = j;
            }
        }
        float best = phases[min_index];

        if (best == best_phase) {
            num_identical++;
            if (num_identical >= precision / 2) {
                break;
            }
        }

        best_phase = best;

        float step = (fmax(phases[0], fmax(phases[1], phases[2])) - fmin(phases[0], fmin(phases[1], phases[2]))) / 2.0f;
        phases[0] = best_phase - step / 2.0f;
        phases[1] = best_phase;
        phases[2] = best_phase + step / 2.0f;

        if (phases[0] < 0.0f || phases[0] > 2.0f * PI) { phases[0] = phases[1]; }
        if (phases[2] < 0.0f || phases[2] > 2.0f * PI) { phases[2] = phases[1]; }

        if (phases[0] == phases[1] && phases[1] == phases[2]) {
            break;
        }
    }

    return best_phase;
}

// Function to find the best amplitude
float find_best_amplitude(device float* residue, float frequency, float phase, uint precision, device int* peak_indices, uint peakCount) {
    float amplitudes[3] = {0.0f, 1.0f, 2.0f};
    float best_amplitude = -1.0f;
    uint num_identical = 0;

    for (uint i = 0; i < precision; ++i) {
        float errors[3] = {0.0f, 0.0f, 0.0f};
        for (uint j = 0; j < 3; ++j) {
            float max_diff = 0.0f;
            for (uint k = 0; k < peakCount; ++k) {
                float val = calculate_sin(frequency, amplitudes[j], phase, float(peak_indices[k]));
                float res = residue[peak_indices[k]];
                max_diff = fmax(max_diff, fabs(val - res));
            }
            errors[j] = max_diff;
        }

        uint min_index = 0;
        for (uint j = 1; j < 3; ++j) {
            if (errors[j] < errors[min_index]) {
                min_index = j;
            }
        }
        float best = amplitudes[min_index];

        if (best_amplitude == best) {
            num_identical++;
            if (num_identical >= precision / 2) {
                break;
            }
        }

        best_amplitude = best;

        float step = (fmax(amplitudes[0], fmax(amplitudes[1], amplitudes[2])) - fmin(amplitudes[0], fmin(amplitudes[1], amplitudes[2]))) / 2.0f;
        amplitudes[0] = best_amplitude - step / 2.0f;
        amplitudes[1] = best_amplitude;
        amplitudes[2] = best_amplitude + step / 2.0f;

        if (amplitudes[0] < 0.0f || amplitudes[0] > 2.0f) { amplitudes[0] = amplitudes[1]; }
        if (amplitudes[2] < 0.0f || amplitudes[2] > 2.0f) { amplitudes[2] = amplitudes[1]; }

        if (amplitudes[0] == amplitudes[1] && amplitudes[1] == amplitudes[2]) {
            break;
        }
    }

    return best_amplitude;
}

// Function to calculate mean
float calculate_mean(device float* numbers, uint size) {
    float sum = 0.0f;
    for (uint i = 0; i < size; ++i) {
        sum += numbers[i];
    }
    return fabs(sum / float(size));
}

// Function to calculate mean of absolute values
float calculate_mean_abs(const device float* numbers, uint size) {
    float sum = 0.0f;
    for (uint i = 0; i < size; ++i) {
        sum += fabs(numbers[i]);
    }
    return sum / float(size);
}

// Kernel function for decompose_sinusoid
kernel void decompose_sinusoid_kernel(const device float* data [[buffer(0)]],
                                      device Sinusoid* sinusoids [[buffer(1)]],
                                      device float* residue [[buffer(2)]],
                                      device float* resultant [[buffer(3)]],
                                      constant float& halving [[buffer(4)]],
                                      constant uint& precision [[buffer(5)]],
                                      constant uint& max_halvings [[buffer(6)]],
                                      constant float& reference_size [[buffer(7)]],
                                      constant float& negligible [[buffer(8)]],
                                      constant uint& sinusoidsCount [[buffer(9)]],
                                      constant uint& peaksCount [[buffer(10)]],
                                      device int* peaks [[buffer(11)]],
                                      uint size [[threads_per_grid]],
                                      uint gid [[thread_position_in_grid]]) {

    float PI = acos(-1.0f);

    uint length = size;
    float relative_frequency = (2.0f * PI) / float(length);
    float frequency = (reference_size <= 1.0f ? reference_size : pow(2.0f, reference_size)) * relative_frequency;

    // Initialize residue with data
    for (uint i = gid; i < length; i += size) {
        residue[i] = data[i];
        resultant[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_device);

    float mean_residue = 1994.0f;
    float prev_frequency = frequency;
    float next_frequency = frequency * 2.0f;
    
    uint sinusoidCount = 0;

    for (uint i = 0; i < max_halvings; ++i) {
        if (calculate_mean_abs(residue, length) < negligible) {
            break;
        }

        // Copy residue to threadgroup memory
        for (uint j = gid; j < length; j += size) {
            residue[j] = residue[j];
        }
        threadgroup_barrier(mem_flags::mem_device);

        // Find peaks for current frequency (using threadgroup memory)
        uint peakCount = find_peaks_kernel(residue, peaks, gid, size, peaksCount);
        threadgroup_barrier(mem_flags::mem_device);

        float amplitude = 1.0f;
        float phase = 0.0f;

        // Find best phase and amplitude
        for (uint j = 0; j < precision / 2; ++j) {
            phase = find_best_phase(residue, frequency, amplitude, precision, peaks, peakCount);
            amplitude = find_best_amplitude(residue, frequency, phase, precision, peaks, peakCount);
        }

        // Generate sinusoid with best parameters
        for (uint j = gid; j < length; j += size) {
            resultant[j] += generate_sinusoid(frequency, amplitude, phase, j);
            residue[j] -= generate_sinusoid(frequency, amplitude, phase, j);
        }
        threadgroup_barrier(mem_flags::mem_device);

        mean_residue = calculate_mean(residue, length);

       // Store the sinusoid parameters (atomically)
        if (amplitude > 0.0f) {
            sinusoids[sinusoidCount].frequency = frequency / relative_frequency;
            sinusoids[sinusoidCount].amplitude = amplitude;
            sinusoids[sinusoidCount].phase = phase;
            sinusoidCount += 1;
            threadgroup_barrier(mem_flags::mem_device);
        }

        prev_frequency = frequency;
        frequency = next_frequency;
        next_frequency *= halving;
    }
    
    threadgroup_barrier(mem_flags::mem_device);
}
