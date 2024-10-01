#include <metal_stdlib>
using namespace metal;

struct Sinusoid {
    float frequency;
    float amplitude;
    float phase;
};

// Helper function to find peaks (using threadgroup memory)
uint find_peaks(device float* residue,
                 device int* peaks,
                 uint id,
                 uint size,
                 uint peaksCount,
                 uint peaksBegin) {

    if (id >= size || size < 2) {
        return 0;
    }

    threadgroup_barrier(mem_flags::mem_device);

    float changes = 0.0f;
    int polarity = 0;
    uint peakCount = peaksBegin;

    for (uint i = 1; i < size; i += 1) {
        float diff = residue[i] - residue[i - 1];
        float absDiff = fabs(diff);
        changes = (changes + absDiff) / 2.0f;  // Update average change
        
        if (absDiff > changes / 2.f) {
            if ((diff > 0.0f && polarity != 1) ||
                (diff < 0.0f && polarity != -1)) {
                peaks[peakCount++] = i - 1;
                polarity = diff > 0.0f ? 1 : -1;
            }
        }

        if (peakCount >= peaksCount) break;
    }

    peaks[peakCount] = -1;
    threadgroup_barrier(mem_flags::mem_device);

    return peakCount;
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
                float diff = res-val;
                diff = pow(diff, 2);
                max_diff += diff;
            }
            errors[j] = max_diff/peakCount;
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
        phases[0] = best_phase - (step / 2.0f);
        phases[1] = best_phase;
        phases[2] = best_phase + (step / 2.0f);

        if (phases[0] < 0.0f) { phases[0] = 0; }
        if (phases[0] > 2.0f * PI) { phases[0] = 2.0f * PI; }
        if (phases[2] < 0.0f) { phases[2] = 0; }
        if (phases[2] > 2.0f * PI) { phases[2] = 2.0f * PI; }

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

    for (uint i = 0; i < precision; i++) {
        float errors[3] = {0.0f, 0.0f, 0.0f};
        for (uint j = 0; j < 3; j++) {
            float max_diff = 0.0f;
            for (uint k = 0; k < peakCount; ++k) {
                float val = calculate_sin(frequency, amplitudes[j], phase, float(peak_indices[k]));
                float res = residue[peak_indices[k]];
                float diff = res-val;
                diff = pow(diff, 2);
                max_diff += diff;
            }
            errors[j] = max_diff/peakCount;
        }

        uint min_index = 0;
        for (uint j = 1; j < 3; j++) {
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
        amplitudes[0] = best_amplitude - (step / 2.0f);
        amplitudes[1] = best_amplitude;
        amplitudes[2] = best_amplitude + (step / 2.0f);

        if (amplitudes[0] < 0.0f) { amplitudes[0] = 0; }
        if (amplitudes[2] < 0.0f) { amplitudes[2] = 0; }

        if (amplitudes[0] == amplitudes[1] && amplitudes[1] == amplitudes[2]) {
            break;
        }
    }

    return best_amplitude;
}

// Function to calculate mean
float calculate_mean(device float* numbers, uint size) {
    float sum = 0.0f;
    for (uint i = 0; i < size; i++) {
        sum += numbers[i];
    }
    return fabs(sum / float(size));
}

float calculate_mean_onPeaks(device float* prev, device float* diff, uint size, device int* peaks, uint peaksCount) {
    float sum = 0.0f;
    for (uint i = 0; i < peaksCount; i++) {
        uint peak = peaks[i];
        float pre = prev[peak];
        float dif = diff[peak];
        float scale = dif-pre;
        if(pre<0) scale *= -1;
        sum += scale;
    }
    return sum / float(size);
}

// Function to calculate mean of absolute values
float calculate_mean_abs(const device float* numbers, uint size) {
    float sum = 0.0f;
    for (uint i = 0; i < size; i++) {
        sum += fabs(numbers[i]);
    }
    return sum / float(size);
}

int compact_peaks(device int* peaks, uint length, uint divisionPoint) {
    if (length <= 1) {
        return length;
    }
    
    uint removed = 0;
    uint readingAt = divisionPoint;
    for(uint i=0; i<divisionPoint; i++){
        int peak = peaks[i];
        
        while(readingAt<length){
            int divPeak = peaks[readingAt];
            
            if(divPeak == peak){
                peaks[readingAt] = -2;
                removed++;
                break;
            }
            
            if(divPeak > peak){
                break;
            }
            
            readingAt++;
        }
    }

    readingAt = divisionPoint;
    uint takeAt = length - 1;
    uint cut = 0;
    while(readingAt <= (length-cut)){
        if(peaks[readingAt] == -2){
            while(true){
                if(takeAt == readingAt)
                    break;
                
                if(peaks[takeAt] != -2){
                    peaks[readingAt] = peaks[takeAt];
                    cut++;
                    takeAt--;
                    readingAt++;
                    break;
                }
                
                takeAt--;
            }
        }
        
        readingAt++;
    }
    
    if(removed > 0)
        peaks[length - removed] = -1;

    return length - removed;
}

///
/// Frequency seeker
///

void generate_sinusoid_and_find_parameters(device float* data,
                                          device float* residue,
                                          device float* resultant,
                                          constant uint& precision,
                                          constant float& negligible,
                                          constant uint& peaksCount,
                                          device int* peaks,
                                          device float* currentSignal,
                                          uint size [[threads_per_grid]],
                                          uint gid [[thread_position_in_grid]],
                                          float frequency,
                                          threadgroup float* tg_amplitude,
                                          threadgroup float* tg_phase,
                                          threadgroup float* tg_mean_diff,
                                          constant int& dataCount) {

    float PI = acos(-1.0f);
    uint length = dataCount;
    float amplitude = 1.0f;
    float phase = 0.0f;

    // Generate sinusoid and find peaks for phase 0
    for (uint j = 0; j < length; j += 1) {
        currentSignal[j] = data[j] - calculate_sin(frequency, amplitude, phase, j);
    }
    threadgroup_barrier(mem_flags::mem_device);

    uint peakCount0 = find_peaks(currentSignal, peaks, gid, length, peaksCount, 0); // Assuming peaksCount is 1024
    threadgroup_barrier(mem_flags::mem_device);

    // Generate sinusoid and find peaks for phase PI
    for (uint j = 0; j < length; j += 1) {
        currentSignal[j] = data[j] - calculate_sin(frequency, amplitude, phase + PI, j);
    }
    threadgroup_barrier(mem_flags::mem_device);

    uint peakCount = find_peaks(currentSignal, peaks, gid, length, peaksCount, peakCount0); // Assuming peaksCount is 1024
    threadgroup_barrier(mem_flags::mem_device);

    // Compact peaks (assuming you have a compact_peaks function)
    peakCount = compact_peaks(peaks, peakCount, peakCount0);

    // Find best phase and amplitude
    float _phase = -1.0f;
    float _amplitude = -1.0f;
    for (uint j = 0; j < precision / 2; j++) {
        phase = find_best_phase(data, frequency, amplitude, precision, peaks, peakCount);
        amplitude = find_best_amplitude(data, frequency, phase, precision, peaks, peakCount);

        if ((fabs(_phase - phase) + fabs(_amplitude - amplitude)) / 2 < negligible) {
            break;
        }

        _phase = phase;
        _amplitude = amplitude;
    }

    // Generate final sinusoid and calculate diff
    float mean_diff = 1994.f;
    
    if (amplitude > negligible){
        for (uint j = 0; j < length; j += 1) {
            float val = calculate_sin(frequency, amplitude, phase, j);
            resultant[j] += val;
            residue[j] = data[j] - val;
        }
        threadgroup_barrier(mem_flags::mem_device);
        
        //mean_diff = calculate_mean(residue, length);
        mean_diff = calculate_mean_onPeaks(data, residue, length, peaks, peakCount);
    }
    else {
        amplitude = 0;
    }

    // Store results in threadgroup memory
    *tg_amplitude = amplitude;
    *tg_phase = phase;
    *tg_mean_diff = mean_diff;
    
    threadgroup_barrier(mem_flags::mem_device);
}

kernel void decompose_sinusoid_adv_kernel(device float* data [[buffer(0)]],
                                      device Sinusoid* sinusoids [[buffer(1)]],
                                      device float* residue [[buffer(2)]], // Temporary residue buffer
                                      device float* resultant [[buffer(3)]], // Temporary resultant buffer
                                      device float* temp_residue [[buffer(4)]], // Temporary residue buffer
                                      device float* temp_resultant [[buffer(5)]], // Temporary resultant buffer
                                      constant float& halving [[buffer(6)]],
                                      constant uint& precision [[buffer(7)]],
                                      constant uint& max_halvings [[buffer(8)]],
                                      constant float& reference_size [[buffer(9)]],
                                      constant float& negligible [[buffer(10)]],
                                      constant uint& sinusoidsCount [[buffer(11)]],
                                      constant uint& peaksCount [[buffer(12)]],
                                      device int* peaks [[buffer(13)]],
                                      device float* currentSignal [[buffer(14)]],
                                      device float* current_resultant [[buffer(15)]],
                                      constant int& dataCount [[buffer(16)]],
                                      uint size [[threads_per_grid]],
                                      uint gid [[thread_position_in_grid]]) { // gid is a stupid thing in this context

    float PI = acos(-1.0f);
    uint length = dataCount;
    float relative_frequency = (2.0f * PI) / float(length);
    float _reference_size = reference_size <= 1.0f ? reference_size : pow(2.0f, reference_size);
    float frequency = _reference_size * relative_frequency;
    float prev_frequency = frequency;
    float next_frequency = frequency * 2.0f;

    float mean_residue = calculate_mean(data, length);
    uint sinusoidIndex = 0;

    for (uint i = 0; i < max_halvings; ++i) {
                
        if(sinusoidIndex >= sinusoidsCount) break;
        
        if (calculate_mean_abs(data, length) < negligible) {
            break;
        }

        threadgroup float tg_amplitude;
        threadgroup float tg_phase;
        threadgroup float tg_mean_diff;

        generate_sinusoid_and_find_parameters(data, residue, resultant, precision, negligible, peaksCount, peaks, currentSignal, size, gid, frequency, &tg_amplitude, &tg_phase, &tg_mean_diff, dataCount);

        float amplitude = tg_amplitude;
        float phase = tg_phase;
        float mean_diff = tg_mean_diff;

        bool no_frequency_halving = false;
        
        if(frequency != prev_frequency){
            float freq = frequency;
            float best_amplitude = 0.0f;
            float best_phase = 0.0f;
            float reference_mean = mean_residue;
            float min_freq = prev_frequency;
            
            for (uint j = 0; j < precision / 2; j++) {
                freq = (freq + min_freq) / 2;
                
                for(uint i=0; i<length; i++){
                    //temp_resultant[i] = current_resultant[i];
                    temp_resultant[i] = 0;
                }
                
                generate_sinusoid_and_find_parameters(data, temp_residue, temp_resultant, precision, negligible, peaksCount, peaks, currentSignal, size, gid, freq, &tg_amplitude, &tg_phase, &tg_mean_diff, dataCount);
                
                if ((tg_mean_diff > mean_diff || tg_mean_diff > reference_mean)) {
                    break;
                }
                
                if (mean_residue - tg_mean_diff < mean_diff - tg_mean_diff) {
                    min_freq = prev_frequency;
                } else {
                    min_freq = frequency;
                }
                
                if (tg_amplitude > 0.0f) {
                    reference_mean = tg_mean_diff;
                    best_amplitude = tg_amplitude;
                    best_phase = tg_phase;
                }
                else {
                    break;
                }
            }
            
            if (best_amplitude > 0.0f) {
                frequency = freq;
                amplitude = best_amplitude;
                phase = best_phase;
                mean_diff = reference_mean;
                no_frequency_halving = true;
                
                for(uint i = 0; i < length; i++){
                    residue[i] = temp_residue[i];
                    resultant[i] = temp_resultant[i];
                }
            }
        }
    

        if (mean_diff > mean_residue) {
            amplitude = 0;
        }

        if (amplitude > 0.0f) {
            mean_residue = mean_diff;
            
            sinusoids[sinusoidIndex].frequency = frequency / relative_frequency;
            sinusoids[sinusoidIndex].amplitude = amplitude;
            sinusoids[sinusoidIndex].phase = phase;
            
            sinusoidIndex++;
            
            for(uint i = 0; i < length; i++){
                data[i] = residue[i];
                current_resultant[i] += resultant[i];
            }
        }
        
        if (no_frequency_halving) {
            prev_frequency = frequency;
            frequency = (frequency + next_frequency)/2; //prev_frequency * halving;
        } else {
            prev_frequency = frequency;
            frequency = next_frequency;
            next_frequency *= halving;
        }
    }
}
