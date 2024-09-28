import numpy as np
from scipy.signal import find_peaks

def find_peaks_iterative(array, n=3):
    """
    Finds the indices of peaks (positive and negative) in an array,
    iteratively searching for smaller peaks.

    Args:
        array: List or array of numbers.
        n: Maximum number of iterations to find smaller peaks.

    Returns:
        A sorted list containing the indices of all peaks.
    """
    positive_peaks = set()
    negative_peaks = set()

    for _ in range(n):
        # Find positive peaks
        peaks, _ = find_peaks(array)  # Uso la funzione di SciPy
        positive_peaks.update(peaks)

        # Find negative peaks by inverting the array
        peaks, _ = find_peaks(-array)  # Uso la funzione di SciPy
        negative_peaks.update(peaks)

        # "Flatten" the found peaks for the next iteration
        for peak in positive_peaks:
            if peak > 0 and peak < len(array) - 1:
                array[peak] = (array[peak-1] + array[peak+1]) / 2
        for peak in negative_peaks:
            if peak > 0 and peak < len(array) - 1:
                array[peak] = (array[peak-1] + array[peak+1]) / 2

    # Combine the indices of positive and negative peaks
    peak_indices = positive_peaks | negative_peaks

    # Convert to list and sort
    peak_indices = sorted(list(peak_indices))

    return peak_indices

def mean_absolute_difference(arr1, arr2):
    """
    Calculates the mean absolute difference between two arrays of numbers.

    Args:
      arr1: The first array.
      arr2: The second array.

    Returns:
      The mean absolute difference.
    """
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must have the same length")
    return np.mean([abs(a - b) for a, b in zip(arr1, arr2)])

def generate_sinusoid(freq, amplitude, phase, length):
    """Generates a sinusoid given the frequency, amplitude, and phase."""
    x = np.arange(length)
    return amplitude * np.sin((x * freq) + phase)

def calculate_sin(freq, amplitude, phase, x):
    return amplitude * np.sin((x * freq) + phase)

def find_best_phase(residue, frequency, amplitude, precision, length, peak_indices):
    """Refines the optimal phase by minimizing the error at peak indices."""
    phases = [0, np.pi, np.pi*1.75]
    for _ in range(precision):
        errors = []
        for phase in phases:
            diffs = []
            for peak in peak_indices:
                val = calculate_sin(frequency, amplitude, phase, peak)
                res = residue[peak]
                diffs.append(abs(val - res))
            errors.append(np.max(diffs))

        best_phase = phases[np.argmin(errors)]

        # Binary search refinement
        step = (max(phases) - min(phases)) / 2
        phases = [best_phase - step / 2, best_phase, best_phase + step / 2]
        phases = [phase for phase in phases if 0 <= phase <= np.pi*2]

        if len(phases) == 1:
            break

    return best_phase

def find_best_amplitude(residue, frequency, phase, precision, length, peak_indices):
    """Refines the optimal amplitude by minimizing the error at peak indices."""
    amplitudes = [0.25, 1, 2]
    for _ in range(precision):
        errors = []
        for amplitude in amplitudes:
            diffs = []
            for peak in peak_indices:
                val = calculate_sin(frequency, amplitude, phase, peak)
                res = residue[peak]
                diffs.append(abs(val - res))
            errors.append(np.max(diffs))

        best_amplitude = amplitudes[np.argmin(errors)]

        # Binary search refinement
        step = (max(amplitudes) - min(amplitudes)) / 2
        amplitudes = [best_amplitude - step / 2, best_amplitude, best_amplitude + step / 2]
        amplitudes = [amp for amp in amplitudes if 0 <= amp <= 2]

        if len(amplitudes) == 1:
            break

    return best_amplitude

def calculate_mean(numbers):
    """Calculates the mean of a list of numbers."""
    return sum(abs(numbers)) / len(numbers)

def union_without_duplicates(list1, list2):
    """Merges two lists without repeating equal elements."""
    return list(set(list1) | set(list2))

def decompose_sinusoid(data, halving=2.0, precision=10, max_halvings=10, reference_size=1):
    length = len(data)
    sinusoids = []
    residue = np.array(data)
    resultant = np.zeros(length)

    frequency = reference_size * ((np.pi*2) / length)
    num_peaks = 3

    mean_residue = calculate_mean(residue)
    for _ in range(max_halvings):
        if mean_residue == 0:
            break

        # Initial amplitude
        amplitude = 1
        phase = 0

        base_wave = generate_sinusoid(frequency, amplitude, phase, length)
        wave = base_wave - residue
        peaks = find_peaks_iterative(wave, num_peaks)

        base_wave = generate_sinusoid(frequency, amplitude, np.pi, length)
        wave = base_wave - residue
        peaks_2 = find_peaks_iterative(wave, num_peaks)

        peaks = union_without_duplicates(peaks, peaks_2)

        for _ in range(0, max(int(precision/3), 1)): # Are not necessary too many precision cycles
            # Find optimal phase
            phase = find_best_phase(residue, frequency, amplitude, precision, length, peaks)

            # Find optimal amplitude
            amplitude = find_best_amplitude(residue, frequency, phase, precision, length, peaks)

        # Generate the optimal sinusoid
        sinusoid = generate_sinusoid(frequency, amplitude, phase, length)

        # Update the residue
        diff = residue - sinusoid

        mean_diff = calculate_mean(diff)
        if abs(mean_diff) > abs(mean_residue): # ignore calculation
            amplitude = 0
        else:
            resultant += sinusoid
            residue = diff
            mean_residue = mean_diff

        if amplitude > 0:
            # Save the current sinusoid
            sinusoids.append({
                'frequency': frequency,
                'phase': phase,
                'amplitude': amplitude
            })

        # Halve the frequency for the next sinusoid
        frequency *= halving

    return sinusoids, residue.tolist(), resultant

# Example usage:
length = 100
refPi = np.pi / (length / 2)
data = [np.sin(refPi * x) + (np.sin((refPi * x * 2) + (np.pi / 4))*0.75) for x in range(length)]

sinusoids, residue, resultant = decompose_sinusoid(data, halving=2.0, precision=10, max_halvings=10, reference_size=1)
print("Sinusoids:", sinusoids)