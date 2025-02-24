<br>

# <p align="center"> Integrated Project for Business

<br><br>

### <p align="center"> GOOD DECISIONS = GOOD RESULTS

<br><br>

### <p align="center"> [![Sponsor Quantum Software Development](https://img.shields.io/badge/Sponsor-Quantum%20Software%20Development-brightgreen?logo=GitHub)](https://github.com/sponsors/Quantum-Software-Development)

<br><br>

## [Standardization of a Range of Values]()

t describes the process of scaling or normalizing data within a specific range, typically to a standardized scale, for example, from 0 to 1. This is a common technique in data analysis and machine learning.

<br>

### **Mathematical Formula**

<br>

$$X_{normalized} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$

<br><br>

### <p align="center"> [Where]():

 $$X_{\max} - X_{\min} = \text{Amplitude}$$ 

 Is the `amplitude`, a way to represent the range of data values before normalization.

 <br>

 ###  Formula for Calculating this Normalized Value in Excel.
 
 ```excel
=(A3-MIN(A$3:A$102))/ (MAX(A$3:A$102)-MIN(A$3:A$102))
```

<br><br>

### **Implementation in Python**

<br>

```python
import numpy as np

def normalize_range(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Example usage:
data = np.array([10, 20, 30, 40, 50])
normalized_data = normalize_range(data)
print(normalized_data)
```

