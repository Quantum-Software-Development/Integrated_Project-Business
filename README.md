<br>

# <p align="center"> Integrated Project for [Business]()

<br><br>

<h3 align="center">  $$\Huge {\textbf{\color{DodgerBlue} GOOD DECISIONS = GOOD RESULTS}}$$ 

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

## [Explanation]():

To calculate the standardization of the variables salario, n_filhos, and idade using both the Z-Score and Range methods, and to evaluate the mean, standard deviation, maximum, and minimum before and after standardization, we can follow these steps:


### [Before Standardization]():

Compute the mean, standard deviation, maximum, and minimum for each of the variables (n_filhos, salario, idade).

#### [Z-Score Standardization]():

We standardize the variables using the Z-Score method, which is computed as:


$Z$ = $\frac{X - \mu}{\sigma}$

```latex
Z = \frac{X - \mu}{\sigma}
```

Where:
- $\( \mu \)$ is the mean,
- $\( \sigma \)$ is the standard deviation.

  <br>

#### [Range Standardization (Min-Max Scaling)]():

We scale the data using the Min-Max method, which scales the values to a [0, 1] range using:

$X'$ = $\frac{X - \min(X)}{\max(X) - \min(X)}$

```latex
X' = \frac{X - \min(X)}{\max(X) - \min(X)}
```
  
Where:
- X is the original value,
- min(X) is the minimum value,
- max(X) is the maximum value.

<br>

### [After Standardization]():

Compute the mean, standard deviation, maximum, and minimum of the standardized data for both Z-Score and Range methods.

The output will provide the descriptive statistics before and after each standardization method, allowing you to compare the effects of Z-Score and Range standardization on the dataset.

 <br>

## Practical Example for Calculating this Normalized Value in [Python]():

#### Use this [dataset](https://github.com/Quantum-Software-Development/Integrated_Project-Business/blob/f2d7abe6ee5853ae29c750170a01e429334f6fe5/HomeWork/1-Z-Score-Range/cadastro_funcionarios.xlsx)

The code demonstrates how to apply Z-Score and Range (Min-Max) standardization to the variables salario, n_filhos, and idade in a dataset. It also evaluates and compares the mean, standard deviation, minimum, and maximum values before and after the standardization methods are applied.

 <br>

#### Cell 1: [Import necessary libraries]()

```python
# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
```

<br>

#### Cell 2: [Load the dataset from the Excel file]()

```python
# Load the data from the Excel file
# df = pd.read_excel('use-your-own-dataset.xlsx') - optional
df = pd.read_excel('cadastro_funcionarios.xlsx')
df.head()  # Displaying the first few rows of the dataset to understand its structure
```

<br>

#### Cell 3: [Evaluate the statistics before standardization]()

```python
# Step 1: Evaluate the mean, std, max, and min before standardization
before_std_stats = {
    'mean_n_filhos': df['n_filhos'].mean(),
    'std_n_filhos': df['n_filhos'].std(),
    'min_n_filhos': df['n_filhos'].min(),
    'max_n_filhos': df['n_filhos'].max(),
    
    'mean_salario': df['salario'].mean(),
    'std_salario': df['salario'].std(),
    'min_salario': df['salario'].min(),
    'max_salario': df['salario'].max(),
    
    'mean_idade': df['idade'].mean(),
    'std_idade': df['idade'].std(),
    'min_idade': df['idade'].min(),
    'max_idade': df['idade'].max(),
}

# Display the statistics before standardization
before_std_stats
```

<br>

Cell 4: [Apply Z-Score standardization]()

```python
# Step 2: Z-Score Standardization
df_zscore = df[['n_filhos', 'salario', 'idade']].apply(lambda x: (x - x.mean()) / x.std())

# Display the standardized data
df_zscore.head()
```

<br>

Cell 5: [Evaluate the statistics after Z-Score standardization]()

```python
# Step 3: Evaluate the mean, std, max, and min after Z-Score standardization
after_zscore_stats = {
    'mean_n_filhos_zscore': df_zscore['n_filhos'].mean(),
    'std_n_filhos_zscore': df_zscore['n_filhos'].std(),
    'min_n_filhos_zscore': df_zscore['n_filhos'].min(),
    'max_n_filhos_zscore': df_zscore['n_filhos'].max(),
    
    'mean_salario_zscore': df_zscore['salario'].mean(),
    'std_salario_zscore': df_zscore['salario'].std(),
    'min_salario_zscore': df_zscore['salario'].min(),
    'max_salario_zscore': df_zscore['salario'].max(),
    
    'mean_idade_zscore': df_zscore['idade'].mean(),
    'std_idade_zscore': df_zscore['idade'].std(),
    'min_idade_zscore': df_zscore['idade'].min(),
    'max_idade_zscore': df_zscore['idade'].max(),
}

# Display the statistics after Z-Score standardization
after_zscore_stats
```

<br>

Cell 6: [Apply Range Standardization]() (Min-Max Scaling)

```python
# Step 4: Range Standardization (Min-Max Scaling)
scaler = MinMaxScaler()
df_range = pd.DataFrame(scaler.fit_transform(df[['n_filhos', 'salario', 'idade']]), columns=['n_filhos', 'salario', 'idade'])

# Display the scaled data
df_range.head()
```

<br>

Cell 7: [Evaluate the statistics after Range standardization]()

```python
# Step 5: Evaluate the mean, std, max, and min after Range standardization
after_range_stats = {
    'mean_n_filhos_range': df_range['n_filhos'].mean(),
    'std_n_filhos_range': df_range['n_filhos'].std(),
    'min_n_filhos_range': df_range['n_filhos'].min(),
    'max_n_filhos_range': df_range['n_filhos'].max(),
    
    'mean_salario_range': df_range['salario'].mean(),
    'std_salario_range': df_range['salario'].std(),
    'min_salario_range': df_range['salario'].min(),
    'max_salario_range': df_range['salario'].max(),
    
    'mean_idade_range': df_range['idade'].mean(),
    'std_idade_range': df_range['idade'].std(),
    'min_idade_range': df_range['idade'].min(),
    'max_idade_range': df_range['idade'].max(),
}

# Display the statistics after Range standardization
after_range_stats
```

<br>




 <br>
 

## Pratical Example for Calculating this Normalized Value in [Excel]() 

#### Use this [dataset](https://github.com/Quantum-Software-Development/Integrated_Project-Business/blob/f2d7abe6ee5853ae29c750170a01e429334f6fe5/HomeWork/1-Z-Score-Range/cadastro_funcionarios.xlsx)

To standardize the variables (salary, number of children, and age) in Excel using the Z-Score and Range methods, you can follow these steps:

 <br>

## I. [Z-Score Standardization]()

### Steps for Z-Score in Excel:

#### 1. [Find the Mean (µ)]():

Use the AVERAGE function to calculate the mean of the column. For example, to find the mean of the salary (column E), use:

```excel
=AVERAGE(E2:E351)
```

<br>

#### 2. [Find the Standard Deviation (σ)]():
   
Use the STDEV.P function to calculate the standard deviation of the column. For example, to find the standard deviation of the salary (column E), use:

```excel
=STDEV.P(E2:E351)
```

<br>

#### 3. [Apply the Z-Score Formula]():

For each value in the column, apply the Z-Score formula. In the first row of the new column, use:

```excel
=(E2 - AVERAGE(E$2:E$351)) / STDEV.P(E$2:E$351)
```

<br>

#### 4.[Drag the formula down to calculate the Z-Score for all the rows]():

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

#### 1. [Find the Min and Max]():

Use the MIN and MAX functions to find the minimum and maximum values of the column. For example, to find the min and max of salary (column E), use:

```excel
=MIN(E2:E351)
=MAX(E2:E351)
```

<br>

#### 2. [Apply the Range Formula]():

For each value in the column, apply the range formula. In the first row of the new column, use:

```excel
=(E2 - MIN(E$2:E$351)) / (MAX(E$2:E$351) - MIN(E$2:E$351))
```

<br>

#### 3.[Drag the formula down to calculate the range standardized values for all the rows]():

Example for Salary:

In cell I2 (new column for range standardized salary), write:

```excel
=(E2 - MIN(E$2:E$351)) / (MAX(E$2:E$351) - MIN(E$2:E$351))
```

Then, drag it down to the rest of the rows.
Repeat the same steps for the variables n_filhos (column D) and idade (column F).

<br>


## Summary of the Process

[Z-Score Standardization]() centers the data around [zero]() and scales it based on the [standard deviation]().

[Range Standardization (Min-Max Scaling)]() rescals the data to a [[0, 1] range]().

Both techniques were applied (given dataset)  to the [columns n_filhos](), [salario](), and [idade](), and the statistics (mean, std, min, max) were calculated before and after the standardization methods.



<br><br>

#

###### <p align="center"> Copyright 2025 Quantum Software Development. Code released under the [MIT License license.](https://github.com/Quantum-Software-Development/Math/blob/3bf8270ca09d3848f2bf22f9ac89368e52a2fb66/LICENSE)




  


