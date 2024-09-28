# Riccardo's transform
A "Fourier" transform made by me.

After a week of intense work, this evening I decided to relax a bit (in my own way). I don't know why I decided to try to create my own Fuorier transform without any mathematical knowledge but just following my intuition (and GeoGebra). As naive as I am, this is the result after 4 hours of work:

[https://github.com/cekkr/riccardo_transform/riccardo_transform.py](https://github.com/cekkr/riccardo_transform/blob/main/riccardo_transform.py)

## How works
In practice it works like this, in a way similar to the various FFTs: you take a frequency, you try various phases, you see which of these sinusoids if subtracted have the lowest peaks and then you do the same procedure with the amplitude. Subtract the obtained sinusoid from the given series of numbers and start again by doubling the frequency (if you set 2, as by default, to the doubling index). The characteristic is that you can start from the frequency you want and deepen the frequencies as many times as you want, regardless of the size of the given array. An obvious limitation is that the way the code is done (look at the find_best_amplitude function), right now, it doesn't work for frequencies that have an amplitude greater than 2.

# Example usage

```python
length = 100
refPi = np.pi / (length / 2)
data = [np.sin(refPi * x) + (np.sin((refPi * x * 2) + (np.pi / 4))*0.75) for x in range(length)]

# Use at least 6 of precision
sinusoids, residue, resultant = decompose_sinusoid(data, halving=2.0, precision=8, max_halvings=10, reference_size=1)
print("Sinusoids:", sinusoids)
```

Results:

```
Sinusoids: [{'frequency': 0.06283185307179587, 'phase': 0, 'amplitude': 1}, {'frequency': 0.12566370614359174, 'phase': 0.7838641826095627, 'amplitude': 0.7470703125}]
```

With halving we mean how much the frequency doubles at each analysis cycle, with precision we mean how deeply we need to check the amplitude and phase (example, if the number to find is 0.3 the algorithm does 0 and 0.5, 0.25, 0.375 ... now that I think about it I have not implemented anything that stops automatically when the result is "extremely precise"), max_halvings and how many times the frequency doubles to look for matches and reference_size is how large the first frequency is with respect to the size of the given array.

## Last changes
- Cycling for n precision find_best_phase and find_best_amplitude gives a great precision improvement
- Code optimization

# Credits

Riccardo Cecchini (rcecchini.ds@gmail.com)