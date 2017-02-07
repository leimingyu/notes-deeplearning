* normalization
* enumerate the categorical data into 1s and 0s for all the options

## split data set
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## confusion matrix
