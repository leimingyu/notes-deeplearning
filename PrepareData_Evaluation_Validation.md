* normalization
* enumerate the categorical data into 1s and 0s for all the options

## split data set
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## confusion matrix
<img src="Figs/confusion-matrix.png" height="250">


## accuracy
<img src="Figs/accuracy01.png" height="200">

## mean absolute error 
<img src="Figs/mse01.png" height="200">

## R2 score
evaluate the linear regression model with the average simplistic model
<img src="Figs/R2score01.png" height="200">
<img src="Figs/R2score02.png" height="200">

## type of errors
* underfitting : too simple, due to bias
* overfitting: too specific for training set, too much variance

## model complexity graph
<img src="Figs/model_complex01.png" height="200">
<img src="Figs/model_complex02.png" height="200">


## cross validation
use KFold in sklearn

http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
