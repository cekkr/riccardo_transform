import numpy as np
from scipy.signal import find_peaks

def find_peaks_iterative(array, n=3):
    """
    Finds the indices of peaks (positive and negative) in an array.
    Iteratively flattens the array around the peaks to identify smaller peaks in subsequent iterations.

    Args:
        array: The input array-like object.
        n: The number of iterations.

    Returns:
        A sorted list of indices of the peaks in the array.
    """

    positive_peaks = set()
    negative_peaks = set()

    for _ in range(n):
        # Find positive and negative peaks using scipy.signal.find_peaks
        peaks, _ = find_peaks(array)
        positive_peaks.update(peaks)

        peaks, _ = find_peaks(-array)
        negative_peaks.update(peaks)

        # Flatten the array around the peaks to prepare for the next iteration
        for peak in positive_peaks:
            if 0 < peak < len(array) - 1:
                array[peak] = (array[peak - 1] + array[peak + 1]) / 2
        for peak in negative_peaks:
            if 0 < peak < len(array) - 1:
                array[peak] = (array[peak - 1] + array[peak + 1]) / 2

    # Combine and sort the peak indices
    peak_indices = positive_peaks | negative_peaks
    return sorted(list(peak_indices))


def generate_sinusoid(freq, amplitude, phase, length):
    """
    Generates a sinusoid wave.

    Args:
        freq: The frequency of the sinusoid.
        amplitude: The amplitude of the sinusoid.
        phase: The phase shift of the sinusoid.
        length: The length of the sinusoid.

    Returns:
        A numpy array representing the sinusoid wave.
    """
    x = np.arange(length)
    return amplitude * np.sin((x * freq) + phase)


def calculate_sin(freq, amplitude, phase, x):
    """Calculates the value of a sinusoid at a given point."""
    return amplitude * np.sin((x * freq) + phase)


def find_best_phase(residue, frequency, amplitude, precision, peak_indices):
    """
    Finds the optimal phase for a sinusoid by minimizing the error at peak indices.

    Args:
        residue: The residual signal after removing previous sinusoids.
        frequency: The frequency of the sinusoid.
        amplitude: The amplitude of the sinusoid.
        precision: The number of iterations for refining the phase.
        peak_indices: The indices of the peaks in the signal.

    Returns:
        The optimal phase.
    """

    phases = [0, np.pi, np.pi * 1.75]  # Initial phases to test
    best_phase = -1
    num_identical = 0

    for _ in range(precision):
        errors = []
        for phase in phases:
            diffs = []
            for peak in peak_indices:
                val = calculate_sin(frequency, amplitude, phase, peak)
                res = residue[peak]
                diffs.append(abs(val - res))
            errors.append(np.max(diffs))  # Use maximum error for robustness

        best = phases[np.argmin(errors)]  # Select phase with minimum error

        if best == best_phase:
            num_identical += 1
            if num_identical >= int(precision / 2):
                break  # Stop if the best phase doesn't change

        best_phase = best

        # Refine the search range using binary search
        step = (max(phases) - min(phases)) / 2
        phases = [best_phase - step / 2, best_phase, best_phase + step / 2]
        phases = [phase for phase in phases if 0 <= phase <= np.pi * 2]  # Ensure phase is within valid range

        if len(phases) == 1:
            break  # Stop if the search range is reduced to a single value

    return best_phase


def find_best_amplitude(residue, frequency, phase, precision, peak_indices):
    """
    Finds the optimal amplitude for a sinusoid by minimizing the error at peak indices.

    Args:
        residue: The residual signal after removing previous sinusoids.
        frequency: The frequency of the sinusoid.
        phase: The phase of the sinusoid.
        precision: The number of iterations for refining the amplitude.
        peak_indices: The indices of the peaks in the signal.

    Returns:
        The optimal amplitude.
    """
    amplitudes = [0, 1, 2]  # Initial amplitudes to test
    best_amplitude = -1
    num_identical = 0

    for _ in range(precision):
        errors = []
        for amplitude in amplitudes:
            diffs = []
            for peak in peak_indices:
                val = calculate_sin(frequency, amplitude, phase, peak)
                res = residue[peak]
                diffs.append(abs(val - res))
            errors.append(np.max(diffs))  # Use maximum error for robustness

        best = amplitudes[np.argmin(errors)]  # Select amplitude with minimum error

        if best_amplitude == best:
            num_identical += 1
            if num_identical >= int(precision / 2):
                break  # Stop if the best amplitude doesn't change

        best_amplitude = best

        # Refine the search range using binary search
        step = (max(amplitudes) - min(amplitudes)) / 2
        amplitudes = [best_amplitude - step / 2, best_amplitude, best_amplitude + step / 2]
        amplitudes = [amp for amp in amplitudes if 0 <= amp <= 2]  # Ensure amplitude is within valid range

        if len(amplitudes) == 1:
            break  # Stop if the search range is reduced to a single value

    return best_amplitude


def calculate_mean(numbers):
    """Calculates the mean of a list of numbers."""
    return np.abs(np.average(numbers))


def union_without_duplicates(list1, list2):
    """Merges two lists while preserving order and avoiding duplicates."""
    combined_list = list1.copy()
    for item in list2:
        if item not in combined_list:
            combined_list.append(item)
    return combined_list


def decompose_sinusoid(data, halving=2.0, precision=6, max_halvings=10, reference_size=1, negligible=0.01):
    """
    Decomposes a given data series into a sum of sinusoids.

    Args:
        data: The input data series.
        halving: The factor by which the frequency is halved in each iteration.
        precision: The number of iterations for refining amplitude and phase.
        max_halvings: The maximum number of times the frequency can be halved.
        reference_size: The initial reference frequency.
        negligible: The threshold for the mean residue below which the decomposition stops.

    Returns:
        A tuple containing:
            - A list of dictionaries, where each dictionary represents a sinusoid with keys 'frequency', 'phase', and 'amplitude'.
            - The residual data after removing the sinusoids.
            - The resultant sum of sinusoids.
    """
    length = len(data)
    sinusoids = []
    residue = np.array(data)
    resultant = np.zeros(length)

    relative_frequency = ((np.pi * 2) / length)
    reference_size = reference_size if reference_size <= 1 else pow(2, reference_size)
    frequency = reference_size * relative_frequency
    num_peaks = 3

    def generate_sin(frequency, data):
        """
        Generates a sinusoid with optimized amplitude and phase for a given frequency and data.

        Args:
            frequency: The frequency of the sinusoid.
            data: The input data series.

        Returns:
            A tuple containing the amplitude, phase, mean difference, difference, and sinusoid.
        """
        amplitude = 1
        phase = 0

        base_wave = generate_sinusoid(frequency, amplitude, phase, length)
        wave = base_wave - data
        peaks = find_peaks_iterative(wave, num_peaks)

        base_wave = generate_sinusoid(frequency, amplitude, np.pi, length)
        wave = base_wave - data
        peaks_2 = find_peaks_iterative(wave, num_peaks)

        peaks = union_without_duplicates(peaks, peaks_2)

        _phase = -1
        _amplitude = -1
        for _ in range(0, int(precision/2)):
            # Find optimal phase and amplitude
            phase = find_best_phase(data, frequency, amplitude, precision, peaks)
            amplitude = find_best_amplitude(data, frequency, phase, precision, peaks)

            if (abs(_phase - phase) + abs(_amplitude - amplitude)) / 2 < negligible:
                break  # Stop if the change in phase and amplitude is negligible

        # Generate the optimal sinusoid
        sinusoid = generate_sinusoid(frequency, amplitude, phase, length)

        # Calculate the difference and mean difference
        diff = data - sinusoid
        mean_diff = calculate_mean(diff)

        return amplitude, phase, mean_diff, diff, sinusoid

    mean_residue = 1994  # Initialize with a large value
    prev_frequency = frequency
    next_frequency = frequency * 2

    for _ in range(max_halvings):
        if np.average(np.abs(residue)) < negligible:
            break  # Stop if the average residue is below the threshold

        # Generate the initial sinusoid
        amplitude, phase, mean_diff, diff, sinusoid = generate_sin(frequency, residue)

        no_frequency_halving = False
        if prev_frequency != frequency:
            freq = frequency
            best_sin = None
            reference_mean = mean_residue
            min_freq = prev_frequency

            # Refine the frequency using binary search
            for _ in range(0, int(precision/2)):
                freq = (freq + min_freq) / 2
                _amplitude, _phase, _mean_diff, _diff, _sinusoid = generate_sin(freq, residue)

                if _mean_diff > mean_diff or _mean_diff > reference_mean:
                    break  # Stop if the mean difference is not improving

                # Adjust the search range based on the mean difference
                if abs(mean_residue - _mean_diff) > abs(mean_diff - _mean_diff):
                    min_freq = frequency
                else:
                    min_freq = frequency / halving

                reference_mean = _mean_diff
                best_sin = [freq, _amplitude, _phase, _mean_diff, _diff, _sinusoid]

            # Update the sinusoid if a better one is found
            if best_sin is not None:
                frequency, amplitude, phase, mean_diff, diff, sinusoid = best_sin
                no_frequency_halving = True

        if mean_diff > mean_residue * 2:  # Ignore the sinusoid if the mean difference is too large
            amplitude = 0
        else:
            # Update the resultant, residue, and mean residue
            resultant += sinusoid
            residue = diff
            mean_residue = mean_diff

        if amplitude > 0:
            # Add the sinusoid to the list
            sinusoids.append({
                'frequency': frequency / relative_frequency,
                'phase': phase,
                'amplitude': amplitude
            })

        # Update the frequencies for the next iteration
        prev_frequency = frequency
        if no_frequency_halving:
            frequency = prev_frequency * halving
        else:
            frequency = next_frequency
            next_frequency *= halving

    return sinusoids, residue.tolist(), resultant

def combine_sinusoids(sinusoids1, sinusoids2, minimum=0.05):
    """
    Combines two arrays of sinusoids into a single array with unique frequencies.

    For sinusoids with the same frequency, it calculates the combined amplitude and phase
    by treating them as complex numbers and summing them. It also handles cases where
    sinusoids with the same frequency might be located at different positions in the input arrays.

    Args:
      sinusoids1: The first array of sinusoids.
      sinusoids2: The second array of sinusoids.
      minimum: The minimum amplitude threshold for a sinusoid to be included in the result.

    Returns:
      A new array of sinusoids with unique frequencies and combined amplitudes and phases.
    """

    combined_sinusoids = {}

    # Process both sinusoid arrays
    for sinusoids in [sinusoids1, sinusoids2]:
        for sinusoid in sinusoids:
            freq = sinusoid['frequency']
            amp = sinusoid['amplitude']
            phase = sinusoid['phase']

            # Convert amplitude and phase to complex number
            if freq in combined_sinusoids:
                existing_amp = combined_sinusoids[freq][0]
                existing_phase = combined_sinusoids[freq][1]

                # Simple check to potentially skip combining very small amplitudes
                if min(existing_amp, amp) / max(existing_amp, amp) < (minimum*2):
                    if amp > existing_amp:
                        combined_sinusoids[freq] = [amp, phase]

                combined_sinusoids[freq] = (amp * np.exp(1j * phase)) + (existing_amp * np.exp(1j * existing_phase))
            else:
                combined_sinusoids[freq] = [amp, phase]  # Store as a list initially

    # Convert back to amplitude and phase
    result = []
    for freq, complex_sinusoid in combined_sinusoids.items():
        if type(complex_sinusoid) is list:  # Check if it's still a list (not combined)
            amplitude = complex_sinusoid[0]
            phase = complex_sinusoid[1]
        else:
            amplitude = np.abs(complex_sinusoid)  # Calculate amplitude from complex number
            phase = np.angle(complex_sinusoid)  # Calculate phase from complex number

        if amplitude > minimum:  # Filter out sinusoids with very small amplitudes
            result.append({
                'frequency': freq,
                'phase': phase,
                'amplitude': amplitude
            })

    result.sort(key=lambda item: item['frequency'])
    return result

# Example usage:
length = 100
refPi = np.pi / (length / 2)
data = [np.sin(refPi * x) + (np.sin((refPi * x * 2) + (np.pi / 4))*0.5) + (np.sin(refPi * x * 3)) + (np.sin(refPi * x * 8)) for x in range(length)]

sinusoids_1, residue, resultant = decompose_sinusoid(data, halving=2, precision=8, max_halvings=10, reference_size=1)
sinusoids_2, residue, resultant = decompose_sinusoid(residue, halving=2, precision=8, max_halvings=10, reference_size=1)

print("Sinusoids 1:", sinusoids_1)
print("Sinusoids 2:", sinusoids_2)
print("Total sinusoids: ", combine_sinusoids(sinusoids_1, sinusoids_2))

'''
Results:

Sinusoids 1: [{'frequency': 1.0, 'phase': 0.07516505860639641, 'amplitude': 0.953125}, {'frequency': 2.0, 'phase': 1.556990499703926, 'amplitude': 0.28125}, {'frequency': 3.0, 'phase': 0.06979612584879667, 'amplitude': 1}, {'frequency': 9.0, 'phase': 5.229340505902151, 'amplitude': 0.0166015625}, {'frequency': 8.0, 'phase': 0, 'amplitude': 1}]
Sinusoids 2: [{'frequency': 1.0, 'phase': 0.013422331893999362, 'amplitude': 0.037109375}, {'frequency': 3.0, 'phase': 4.344233591292136, 'amplitude': 0.04296875}, {'frequency': 4.0, 'phase': 1.1167380135807468, 'amplitude': 0.017578125}, {'frequency': 8.0, 'phase': 5.014583195598162, 'amplitude': 0.005859375}, {'frequency': 10.0, 'phase': 0.40803888957758055, 'amplitude': 0.005859375}]
Total sinusoids:  [{'frequency': 1.0, 'phase': np.float64(0.07285253743630027), 'amplitude': np.float64(0.990166311532301)}, {'frequency': 2.0, 'phase': 1.556990499703926, 'amplitude': 0.28125}, {'frequency': 3.0, 'phase': np.float64(0.030181145777831077), 'amplitude': np.float64(0.9825484677073023)}, {'frequency': 8.0, 'phase': np.float64(-0.0055840659740489), 'amplitude': np.float64(1.0017594603231335)}]
'''