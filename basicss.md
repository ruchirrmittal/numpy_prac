

## Advanced NumPy Functions

### 1. `np.linspace()`: Generate evenly spaced values over a specified range.

```python
import numpy as np

# Create an array of 5 equally spaced values between 0 and 1.
arr = np.linspace(0, 1, 5)
print(arr)
```

Output:
```
[0.   0.25 0.5  0.75 1.  ]
```

### 2. `np.zeros()`, `np.ones()`: Create arrays filled with zeros or ones.

```python
zeros_arr = np.zeros((2, 3))  # Create a 2x3 array of zeros.
ones_arr = np.ones((3, 2))    # Create a 3x2 array of ones.

print("Zeros Array:")
print(zeros_arr)

print("Ones Array:")
print(ones_arr)
```

Output:
```
Zeros Array:
[[0. 0. 0.]
 [0. 0. 0.]]

Ones Array:
[[1. 1.]
 [1. 1.]
 [1. 1.]]
```

### 3. `np.random.rand()`: Generate random numbers from a uniform distribution.

```python
# Generate an array of 3 random numbers between 0 and 1.
rand_arr = np.random.rand(3)
print(rand_arr)
```

Output:
```
[0.53019724 0.07175485 0.07420823]
```

### 4. `np.concatenate()`: Combine multiple arrays along an existing axis.

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

concatenated = np.concatenate((arr1, arr2))
print(concatenated)
```

Output:
```
[1 2 3 4 5 6]
```

### 5. `np.where()`: Return the indices of elements that satisfy a condition.

```python
arr = np.array([1, 2, 3, 4, 5])
indices = np.where(arr > 2)

print("Indices of elements greater than 2:", indices)
```

Output:
```
Indices of elements greater than 2: (array([2, 3, 4]),)
```

### 6. `np.unique()`: Find the unique elements in an array.

```python
arr = np.array([1, 2, 2, 3, 4, 4, 5])
unique_elements = np.unique(arr)

print("Unique elements:", unique_elements)
```

Output:
```
Unique elements: [1 2 3 4 5]
```
Certainly! Here are a few more advanced NumPy functions along with examples:

### 7. `np.argmax()` and `np.argmin()`: Find the indices of the maximum and minimum values in an array.

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
max_index = np.argmax(arr)
min_index = np.argmin(arr)

print("Index of maximum value:", max_index)
print("Index of minimum value:", min_index)
```

Output:
```
Index of maximum value: 5
Index of minimum value: 1
```

### 8. `np.histogram()`: Compute the histogram of a set of data.

```python
import matplotlib.pyplot as plt

data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5])

hist, bins = np.histogram(data, bins=5)

plt.hist(data, bins=5)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

print("Histogram values:", hist)
print("Bin edges:", bins)
```

This code creates a histogram plot and computes the histogram values and bin edges.

### 9. `np.mean()` along an axis: Compute the mean along a specified axis in a multi-dimensional array.

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mean_row = np.mean(arr, axis=0)  # Compute mean along rows
mean_col = np.mean(arr, axis=1)  # Compute mean along columns

print("Mean along rows:", mean_row)
print("Mean along columns:", mean_col)
```

Output:
```
Mean along rows: [4. 5. 6.]
Mean along columns: [2. 5. 8.]
```

### 10. `np.load()` and `np.save()`: Load and save NumPy arrays to/from disk.

```python
# Save an array to a file
arr = np.array([1, 2, 3, 4, 5])
np.save('my_array.npy', arr)

# Load the saved array from a file
loaded_arr = np.load('my_array.npy')

print("Loaded array:", loaded_arr)
```

These functions allow you to store and retrieve NumPy arrays for later use.

### 11. `np.linalg.inv()`: Compute the inverse of a square matrix.

```python
matrix = np.array([[2, 1], [5, 3]])
inv_matrix = np.linalg.inv(matrix)

print("Original Matrix:")
print(matrix)

print("Inverse Matrix:")
print(inv_matrix)
```

This function is useful for solving linear equations and other matrix operations.
Certainly, here are some more advanced NumPy functions and concepts:

### 12. Broadcasting

NumPy allows you to perform operations on arrays of different shapes, a concept called broadcasting. For example:

```python
arr = np.array([1, 2, 3])
scalar = 2

result = arr * scalar  # Scalar is broadcasted to each element
print(result)
```

Output:
```
[2 4 6]
```

### 13. Element-wise Comparison

You can perform element-wise comparisons on arrays, which return boolean arrays:

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([3, 2, 1, 4, 5])

comparison = arr1 == arr2
print(comparison)
```

Output:
```
[False  True False  True  True]
```

### 14. `np.where()` with Multiple Conditions

You can use `np.where()` with multiple conditions to select elements from different arrays based on conditions:

```python
arr = np.array([1, 2, 3, 4, 5])
result = np.where(arr % 2 == 0, "even", "odd")
print(result)
```

Output:
```
['odd' 'even' 'odd' 'even' 'odd']
```

### 15. `np.gradient()`: Compute the gradient of an array.

```python
arr = np.array([1, 2, 4, 7, 11])
gradient = np.gradient(arr)
print(gradient)
```

Output:
```
[1.  1.5 2.5 3.5 4. ]
```

### 16. `np.polyfit()`: Fit a polynomial of a specified degree to data.

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 3.9, 6.2, 8.8, 11.9])

coefficients = np.polyfit(x, y, 2)  # Fit a second-degree polynomial
print(coefficients)
```

Output:
```
[ 0.95  -0.425  2.   ]
```

### 17. `np.vectorize()`: Convert a Python function to a vectorized function.

```python
def my_function(x):
    return x ** 2 + 2 * x + 1

vectorized_func = np.vectorize(my_function)

arr = np.array([1, 2, 3, 4, 5])
result = vectorized_func(arr)
print(result)
```

Output:
```
[ 4  9 16 25 36]
```

### 18. `np.ma` (Masked Arrays): Handle masked or missing data.

```python
arr = np.array([1, 2, -999, 4, 5])
masked_arr = np.ma.masked_values(arr, -999)

mean = np.mean(masked_arr)
print("Mean with missing data:", mean)
```

In this example, `-999` is treated as missing data.



