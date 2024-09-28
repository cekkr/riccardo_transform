# Riccardo's transform
A "Fourier" transform made by me.

After a week of intense work, this evening I decided to relax a bit (in my own way). I don't know why I decided to try to create my own Fuorier transform without any mathematical knowledge but just following my intuition (and GeoGebra). As naive as I am, this is the result after 4 hours of work:

[https://github.com/cekkr/riccardo_transform/riccardo_transform.py](https://github.com/cekkr/riccardo_transform/blob/main/riccardo_transform.py)

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

## Last changes
- Cycling for n precision find_best_phase and find_best_amplitude gives a great precision improvement
- Code optimization

# Credits

Riccardo Cecchini (rcecchini.ds@gmail.com)