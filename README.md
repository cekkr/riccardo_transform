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

data = [np.sin(refPi * x) + (np.sin((refPi * x * 2) + (np.pi / 4))*0.5) + (np.sin(refPi * x * 3)) for x in range(length)]

sinusoids, residue, resultant = decompose_sinusoid(data, halving=2.0, precision=8, max_halvings=10, reference_size=1)
print("Sinusoids:", sinusoids)
```

Results:

```
Sinusoids: [{'frequency': 1.0, 'phase': 0.19462381246299076, 'amplitude': 1}, {'frequency': 2.0, 'phase': 0.7972865145035621, 'amplitude': 0.5277099609375}, {'frequency': 3.0, 'phase': 0, 'amplitude': 0.97265625}, {'frequency': 6.0, 'phase': 5.25350070331135, 'amplitude': 0.0321044921875}, {'frequency': 7.0, 'phase': 4.59580644050538, 'amplitude': 0.0321044921875}, {'frequency': 14.0, 'phase': 4.81324821718817, 'amplitude': 0.0321044921875}, {'frequency': 15.0, 'phase': 4.516039439535327, 'amplitude': 0.0321044921875}, {'frequency': 30.0, 'phase': 4.270986008670596, 'amplitude': 0.0321044921875}, {'frequency': 32.0, 'phase': 0.0859029241215959, 'amplitude': 0.0321044921875}, {'frequency': 48.0, 'phase': 5.760864848904525, 'amplitude': 0.0321044921875}]
```

With halving we mean how much the frequency doubles at each analysis cycle, with precision we mean how deeply we need to check the amplitude and phase (example, if the number to find is 0.3 the algorithm does 0 and 0.5, 0.25, 0.375 ... now that I think about it I have not implemented anything that stops automatically when the result is "extremely precise"), max_halvings and how many times the frequency doubles to look for matches and reference_size is how large the first frequency is with respect to the size of the given array.

# Example of double cycling and combining

```python
length = 100
refPi = np.pi / (length / 2)
data = [np.sin(refPi * x) + (np.sin((refPi * x * 2) + (np.pi / 4))*0.5) + (np.sin(refPi * x * 3)) for x in range(length)]

sinusoids_1, residue, resultant = decompose_sinusoid(data, halving=2, precision=10, max_halvings=10, reference_size=1)
sinusoids_2, residue, resultant = decompose_sinusoid(residue, halving=2, precision=10, max_halvings=10, reference_size=1)

print("Sinusoids 1:", sinusoids_1)
print("Sinusoids 2:", sinusoids_2)
print("Total sinusoids: ", combine_sinusoids(sinusoids_1, sinusoids_2))
```

Results:

```
Sinusoids 1: [{'frequency': 1.0, 'phase': 0.19462381246299076, 'amplitude': 1}, {'frequency': 2.0, 'phase': 0.7972865145035621, 'amplitude': 0.53125}, {'frequency': 3.0, 'phase': 0, 'amplitude': 0.96875}, {'frequency': 9.0, 'phase': 0.23220634176618896, 'amplitude': 0.0078125}, {'frequency': 8.0, 'phase': 0.01845570635424912, 'amplitude': 0.00286865234375}, {'frequency': 17.0, 'phase': 4.858884145627767, 'amplitude': 0.00286865234375}]

Sinusoids 2: [{'frequency': 1.0, 'phase': 4.6709714991117774, 'amplitude': 0.2509765625}, {'frequency': 2.0, 'phase': 4.575097699868925, 'amplitude': 0.09375}, {'frequency': 4.0, 'phase': 0.10737865515199488, 'amplitude': 0.03125}, {'frequency': 13.0, 'phase': 3.141592653589793, 'amplitude': 0.0078125}, {'frequency': 44.99999999999999, 'phase': 5.691068723055729, 'amplitude': 0.00146484375}]

Total sinusoids:  [{'frequency': 1.0, 'phase': np.float64(-0.059024974139288255), 'amplitude': np.float64(0.9724220892929857)}, {'frequency': 2.0, 'phase': np.float64(0.6756928770880679), 'amplitude': np.float64(0.4592330500096089)}, {'frequency': 3.0, 'phase': 0, 'amplitude': 0.96875}, {'frequency': 4.0, 'phase': 0.10737865515199488, 'amplitude': 0.03125}]
```

## Last changes
- Cycling for n precision find_best_phase and find_best_amplitude gives a great precision improvement
- Code optimization
- Relative frequencies
- Double cycling decomposition and combine sinusoids

# Credits

Riccardo Cecchini (rcecchini.ds@gmail.com)