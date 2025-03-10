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


## Important Notes

- **Correlation does not imply causation**: Correlation between two variables does not necessarily mean that one causes the other. For example, there may be a correlation between the number of salespeople in a store and increased sales, but that does not imply that having more salespeople directly causes higher sales.

- **Using regressions we don’t need to worry about standardization**: When using regressions, there is no need to worry about data standardization. Unlike other methods like k-NN or neural networks, where the scale of the data can impact performance, regression models can be applied directly without the need for scaling adjustments.

## Pearson Correlation

**Pearson Correlation** is a statistical measure that describes the strength and direction of a linear relationship between two variables. The Pearson correlation value ranges from -1 to 1:

- **1**: Perfect positive correlation (both variables increase together).
- **-1**: Perfect negative correlation (one variable increases while the other decreases).
- **0**: No linear correlation.

For example, if we're analyzing the correlation between the area of a house and its price, a Pearson value close to 1 would indicate that as the area increases, the price also tends to increase.

## Simple Linear Regression

**Simple Linear Regression** is a statistical model that describes the relationship between a dependent variable (response) and an independent variable (predictor). The model is represented by the formula:

$$
y = \beta_0 + \beta_1 \cdot x
$$

Where:
- \(y\) is the dependent variable (the one we want to predict),
- \(x\) is the independent variable (the one used to make predictions),
- \(\beta_0\) is the intercept (the value of \(y\) when \(x = 0\)),
- \(\beta_1\) is the coefficient (representing the change in \(y\) when \(x\) increases by one unit).

Simple linear regression is widely used for predicting a value based on a linear relationship between variables.

### Steps to Perform Linear Regression:

1. **Data Collection**: Gather the data related to the dependent and independent variables.
2. **Exploratory Data Analysis (EDA)**: Explore the data to identify trends, patterns, and check correlation.
3. **Model Fitting**: Fit the linear regression model to the data using a method like Ordinary Least Squares (OLS).
4. **Model Evaluation**: Evaluate the model performance using metrics like Mean Squared Error (MSE) and the Coefficient of Determination (\(R^2\)).
5. **Prediction**: Use the fitted model to make predictions with new data.

Simple linear regression is a great starting point for predictive problems where a linear relationship between variables is expected.


### I- Example Code - [Correlation Vendas Gjornal]()

### Use This Dataset - [BD Gerais.xlsx](https://github.com/Quantum-Software-Development/Integrated_Project-Business/blob/4331d9227118d2025a6c167a3cefd99bf7404939/class_2-Linear%20Regression/BD%20Gerais.xlsx)



### Step 1: Install Required Libraries

If you don't have the required libraries installed, you can install them with pip:

```python
pip install pandas numpy matplotlib scikit-learn openpyxl
```

<br>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from the Excel file
file_path = 'BD Gerais.xlsx'
df = pd.read_excel(file_path)

# Display the first few rows of the dataset
print(df.head())

# Let's assume the columns are: 'Vendas', 'Gjornal', 'GTV', 'Gmdireta'
# Compute the correlation matrix
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Perform linear regression: Let's use 'Vendas' as the target and 'Gjornal', 'GTV', 'Gmdireta' as features
X = df[['Gjornal', 'GTV', 'Gmdireta']]  # Features
y = df['Vendas']  # Target variable

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Print the regression coefficients
print("\nRegression Coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Make predictions
y_pred = model.predict(X)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot the actual vs predicted values
plt.scatter(y, y_pred)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
plt.title('Actual vs Predicted Vendas')
plt.xlabel('Actual Vendas')
plt.ylabel('Predicted Vendas')
plt.show()

# Plot the regression line for each feature vs 'Vendas'
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot for 'Gjornal'
axs[0].scatter(df['Gjornal'], y, color='blue')
axs[0].plot(df['Gjornal'], model.intercept_ + model.coef_[0] * df['Gjornal'], color='red')
axs[0].set_title('Gjornal vs Vendas')
axs[0].set_xlabel('Gjornal')
axs[0].set_ylabel('Vendas')

# Plot for 'GTV'
axs[1].scatter(df['GTV'], y, color='blue')
axs[1].plot(df['GTV'], model.intercept_ + model.coef_[1] * df['GTV'], color='red')
axs[1].set_title('GTV vs Vendas')
axs[1].set_xlabel('GTV')
axs[1].set_ylabel('Vendas')

# Plot for 'Gmdireta'
axs[2].scatter(df['Gmdireta'], y, color='blue')
axs[2].plot(df['Gmdireta'], model.intercept_ + model.coef_[2] * df['Gmdireta'], color='red')
axs[2].set_title('Gmdireta vs Vendas')
axs[2].set_xlabel('Gmdireta')
axs[2].set_ylabel('Vendas')

plt.tight_layout()
plt.show()
````

#

### II- Example Code - [Correlation Vendas -  GTV]()

### Use This Dataset - [BD Gerais.xlsx](https://github.com/Quantum-Software-Development/Integrated_Project-Business/blob/4331d9227118d2025a6c167a3cefd99bf7404939/class_2-Linear%20Regression/BD%20Gerais.xlsx)

To compute the correlation between the Vendas and GTV columns in your dataset using Python, you can follow this code. This will calculate the correlation coefficient and visualize the relationship between these two variables using a scatter plot.

```python
import pandas as pd
import matplotlib.pyplot as plt
```
  
<br><br>

### II - Multiple Linear Regression with 4 variable 

- Vendas as the dependent variable (Y)
  
- Jornal, GTV, and Gmdireta as independent variables (X)
  
This code will also calculate the correlation matrix, fit the multiple linear regression model, and display the regression results.

#### Python Code for Multiple Linear Regression and Correlation

<br>

#### 1- Install Required Libraries (if you don't have them yet)

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn
```

<br>

#### 2- Python Code

<br>

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from the Excel file
file_path = 'BD Gerais.xlsx'  # Adjust the file path if needed
df = pd.read_excel(file_path)

# Display the first few rows of the dataset to verify the data
print(df.head())

# Calculate the correlation matrix for the variables
correlation_matrix = df[['Vendas', 'Gjornal', 'GTV', 'Gmdireta']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Define the independent variables (X) and the dependent variable (Y)
X = df[['Gjornal', 'GTV', 'Gmdireta']]  # Independent variables
y = df['Vendas']  # Dependent variable (Vendas)

# Add a constant (intercept) to the independent variables
X = sm.add_constant(X)

# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# Display the regression results
print("\nRegression Results:")
print(model.summary())

# Alternatively, using sklearn's LinearRegression to calculate the coefficients and R-squared
model_sklearn = LinearRegression()
model_sklearn.fit(X[['Gjornal', 'GTV', 'Gmdireta']], y)

# Coefficients and intercept
print("\nLinear Regression Coefficients (sklearn):")
print("Intercept:", model_sklearn.intercept_)
print("Coefficients:", model_sklearn.coef_)

# Predicting with the model
y_pred = model_sklearn.predict(X[['Gjornal', 'GTV', 'Gmdireta']])

# Calculating R-squared and Mean Squared Error (MSE)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"\nR-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# Plotting the actual vs predicted Vendas
plt.scatter(y, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k', color='red')  # line of perfect prediction
plt.xlabel('Actual Vendas')
plt.ylabel('Predicted Vendas')
plt.title('Actual vs Predicted Vendas')
plt.show()
```

<br>

#### Code Explanation

Loading Data:

The dataset is loaded from BD Gerais.xlsx using pandas.read_excel(). The file path is adjusted based on your actual file location.
Correlation Matrix:

We calculate the correlation matrix for the four variables: Vendas, Gjornal, GTV, and Gmdireta. This gives us an overview of the relationships between the variables.

#### Multiple Linear Regression:

We define the independent variables (Gjornal, GTV, Gmdireta) as X and the dependent variable (Vendas) as y.
We add a constant term (intercept) to X using sm.add_constant() for proper regression.
We use the statsmodels.OLS method to fit the multiple linear regression model and print the regression summary, which includes coefficients, R-squared, p-values, and more.
Alternative Model (sklearn):

We also use sklearn.linear_model.LinearRegression() for comparison, which calculates the coefficients and R-squared.
We then use the trained model to predict the Vendas values and calculate Mean Squared Error (MSE) and R-squared.
Plotting:

The actual values of Vendas are plotted against the predicted values from the regression model in a scatter plot. A red line of perfect prediction is also added (this line represents the ideal case where actual values equal predicted values).

#### Output of the Code:

Correlation Matrix:

Displays the correlation between Vendas, Gjornal, GTV, and Gmdireta. This helps you understand the relationships between these variables.
Regression Results (from statsmodels):

The regression summary will include:
[Coefficients](): The relationship between each independent variable and the dependent variable (Vendas).
[R-squared](): Measures how well the model fits the data.
[P-values](): For testing the statistical significance of each coefficient.

#### Linear Regression Coefficients:

- The model's intercept and coefficients are printed for comparison.

- R-squared and Mean Squared Error (MSE):

- These two metrics evaluate the performance of the regression model.

- R-squared tells you how well the model explains the variance in the dependent variable.
  
- MSE gives an idea of the average squared difference between the predicted and actual values.

  #
  
#### Plot:

The plot shows how well the model's predicted Vendas values match the actual values.


#### Example Output (Model Summary from statsmodels):

<b>

```plaintext
                            OLS Regression Results
==============================================================================
Dep. Variable:                 Vendas   R-squared:                       0.982
Model:                            OLS   Adj. R-squared:                  0.980
Method:                 Least Squares   F-statistic:                     530.8
Date:                Thu, 10 Mar 2025   Prob (F-statistic):           2.31e-14
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         12.1532      3.001      4.055      0.001       5.892      18.414
Gjornal        2.4503      0.401      6.100      0.000       1.638       3.262
GTV            1.2087      0.244      4.948      0.000       0.734       1.683
Gmdireta       0.5003      0.348      1.437      0.168      -0.190       1.191
==============================================================================
```

#### In the summary:

R-squared of 0.982 indicates that the model explains 98.2% of the variance in `Vendas`.

The coefficients of the independent variables show how each variable affects `Vendas`.

<br>

If the p-value for `Gmdireta` is greater than 0.05, it means that `Gmdireta` is not statistically significant in explaining the variability in the dependent variable `Vendas`. In such a case, it's common practice to remove the variable from the model and perform the regression again with only the statistically significant variables.

#### In this case, [you can exclude Gmdireta and rerun the regression model using only the remaining variables](): `Gjornal` and `GTV`.

<br>

#### [Why Remove Gmdireta]()?

[P-value](): The `p-value` is used to test the null hypothesis that the coefficient of the variable is equal to zero (i.e., the variable has no effect). If the p-value is greater than `0.05,` it indicates that the variable is not statistically significant at the 5% level and doesn't provide much explanatory power in the model.

[Adjusted R-squared](): By removing variables that are not significant, you often improve the model's explanatory power (in some cases), as it helps reduce multicollinearity and overfitting.






<br><br>

#

###### <p align="center"> Copyright 2025 Quantum Software Development. Code released under the [MIT License license.](https://github.com/Quantum-Software-Development/Math/blob/3bf8270ca09d3848f2bf22f9ac89368e52a2fb66/LICENSE)




  


