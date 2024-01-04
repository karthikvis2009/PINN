import numpy as np
import timeit

# Create a 1D array
arr = np.arange(1000000)

# Timing np.roll
def roll_operation():
    np.roll(arr, shift=3)

roll_time = timeit.timeit(roll_operation, number=1000)

# Timing array slicing
def slice_operation():
    arr[:-3]

slice_time = timeit.timeit(slice_operation, number=1000)

print(f"np.roll time: {roll_time:.5f} seconds")
print(f"Array slicing time: {slice_time:.5f} seconds")
import numpy as np
import timeit

# Create a 1D array
arr = np.arange(1000000)

# Timing np.roll
def roll_operation():
    np.roll(arr, shift=3)

roll_time = timeit.timeit(roll_operation, number=1000)

# Timing array slicing
def slice_operation():
    arr[:-3]

slice_time = timeit.timeit(slice_operation, number=1000)

print(f"np.roll time: {roll_time:.5f} seconds")
print(f"Array slicing time: {slice_time:.5f} seconds")
