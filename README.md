<br>

# <p align="center"> Integrated Project for Business

<br><br>

### <p align="center"> GOOD DECISIONS = GOOD RESULTS

<br><br>

### <p align="center"> [![Sponsor Quantum Software Development](https://img.shields.io/badge/Sponsor-Quantum%20Software%20Development-brightgreen?logo=GitHub)](https://github.com/sponsors/Quantum-Software-Development)

<br>

## [Standardization of a Range of Values]()

t describes the process of scaling or normalizing data within a specific range, typically to a standardized scale, for example, from 0 to 1. This is a common technique in data analysis and machine learning.

<br>

###  <p align="center"> [Mathematical Formula]()

<br>

$$X_{normalized} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$

<br>

### <p align="center"> [Where]():

 $$X_{\max} - X_{\min} = \text{Amplitude}$$ 

 <br>

####  <p align="center"> Is the `amplitude`, a way to represent the range of data values before normalization.

 <br>


## [Practical Example in Python]():

### Use this [dataset]().

The code demonstrates how to apply Z-Score and Range (Min-Max) standardization to the variables salario, n_filhos, and idade in a dataset. It also evaluates and compares the mean, standard deviation, minimum, and maximum values before and after the standardization methods are applied.

To calculate the standardization of the variables salario, n_filhos, and idade using both the Z-Score and Range methods, and to evaluate the mean, standard deviation, maximum, and minimum before and after standardization, we can follow these steps:

 <br>

## Pratical Example in Excel - [Formula for Calculating this Normalized Value in Excel]()

### Use this [dataset](https://github.com/Quantum-Software-Development/Integrated_Project-Business/blob/f2d7abe6ee5853ae29c750170a01e429334f6fe5/HomeWork/1-Z-Score-Range/cadastro_funcionarios.xlsx)

To standardize the variables (salary, number of children, and age) in Excel using the Z-Score and Range methods, you can follow these steps:

 <br>

## I. [Z-Score Standardization]()

### Steps for Z-Score in Excel:

### 1. [Find the Mean (µ)]():

Use the AVERAGE function to calculate the mean of the column. For example, to find the mean of the salary (column E), use:

```excel
=AVERAGE(E2:E351)
```

<br>

### 2. [Find the Standard Deviation (σ)]():
   
Use the STDEV.P function to calculate the standard deviation of the column. For example, to find the standard deviation of the salary (column E), use:

```excel
=STDEV.P(E2:E351)
```

<br>

### 3. [Apply the Z-Score Formula]():

For each value in the column, apply the Z-Score formula. In the first row of the new column, use:

```excel
=(E2 - AVERAGE(E$2:E$351)) / STDEV.P(E$2:E$351)
```

<br>

### 4.[Drag the formula down to calculate the Z-Score for all the rows]():

Example for Salary:

In cell H2 (new column for standardized salary), write

```excel
=(E2 - AVERAGE(E$2:E$351)) / STDEV.P(E$2:E$351)
```

Then, drag it down to the rest of the rows.

Repeat the same steps for the variables n_filhos (column D) and idade (column F).


<br>

## II. [Range Standardization]()

Steps for Range Standardization in Excel:

### 1. [Find the Min and Max]():

Use the MIN and MAX functions to find the minimum and maximum values of the column. For example, to find the min and max of salary (column E), use:

```excel
=MIN(E2:E351)
=MAX(E2:E351)
```

<br>

### 2. [Apply the Range Formula]():

For each value in the column, apply the range formula. In the first row of the new column, use:

```excel
=(E2 - MIN(E$2:E$351)) / (MAX(E$2:E$351) - MIN(E$2:E$351))
```

<br>

### 3.[Drag the formula down to calculate the range standardized values for all the rows]():

Example for Salary:

In cell I2 (new column for range standardized salary), write:

```excel
=(E2 - MIN(E$2:E$351)) / (MAX(E$2:E$351) - MIN(E$2:E$351))
```

Then, drag it down to the rest of the rows.
Repeat the same steps for the variables n_filhos (column D) and idade (column F).

<br><br>
















<!--
### [Implementation in Python]()

```python
import numpy as np

def normalize_range(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Example usage:
data = np.array([10, 20, 30, 40, 50])
normalized_data = normalize_range(data)
print(normalized_data)
```

<br>

This function takes an array of numerical values and normalizes them within the range [0,1].

<br><br>

## [General Notes]()

- This normalization technique is commonly used in **machine learning preprocessing** to ensure features have the same scale.
- If you want to **normalize to a custom range** \[a, b]\, the formula is:
  
  $$X_{scaled} = a + \left(\frac{X - X_{\min}}{X_{\max} - X_{\min}}\right) \times (b - a)$$

<br>
 
  ### [Python implementation]():
  
  ```python
  def normalize_custom_range(data, a, b):
      return a + ((data - np.min(data)) / (np.max(data) - np.min(data))) * (b - a)
  ```

-->



<br><br>

#

###### <p align="center"> Copyright 2025 Quantum Software Development. Code released under the [MIT License license.](https://github.com/Quantum-Software-Development/Math/blob/3bf8270ca09d3848f2bf22f9ac89368e52a2fb66/LICENSE)




  


