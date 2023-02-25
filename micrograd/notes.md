# Notes on micrograd

- [Notes on micrograd](#notes-on-micrograd)
  - [Example usage](#example-usage)
  - [What micrograd is built on](#what-micrograd-is-built-on)
  - [What is a derivative?](#what-is-a-derivative)

## Example usage

Neural networks are just mathematical expressions

```python
from micrograd.engine import Value

# two inputs
# Value() object wraps numbers
# scalar values are understanding (in production, it's tensors)
a = Value(-4.0)
b = Value(2.0)

# build math expression where a and b are transformed

# here child nodes of c are a and b (pointers)
c = a + b # addition
d = a * b + b**3 # multiplication, exponentiation
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2 # exponentiation

# output value
g = f / 2.0 # division
g += 10.0 / f

# .data is forward pass (value of g)
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass

# backprop at node g
# what? start at g, go backwards through expression graph, recursively apply chain rule
# apply derivative of g with respect to all nodes
g.backward()

# query derivative of g with respect to a and b
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db

# interpretation
# 138.8338 = if we nudge a, g will grow by 138.8338
# how g will respond if a and b are tweaked by a tiny amount
```

## What micrograd is built on

- `engine.py` - 100 lines of code for backprop (doesn't care about nn)
- `nn.py` - 50 lines of code for defining neuron, layer, and MLP

## What is a derivative?

$L = lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$

if you slightly bump a by h, how does the function respond (with what sensitivity)? Does it go down or up and by how much? That's the slope of the function

Example function

$f(x) = 3x^2 - 4x + 5 = 6x - 4$

```py
h = 0.00000001
x = 3.0
f(x + h) # do you expect function to be greater or less after bumping by h?
(f(x + h) - f(x)) / h # tells us how function responded in positive direction normalized by run
```

Look at the [notebook](derivatives.ipynb) for more
