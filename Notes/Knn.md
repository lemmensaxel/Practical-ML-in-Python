## K Nearest Neighbors 

- The goal of a classification algorithm is to create a model that best divides/seperates data. 
- We know that some datapoints are a "+", and some are a "-". With a classificatio algorithm one can classify an unknown data point as either a "+" or a "-".
- The KNN algorithm uses proximity as a qualifier.
- The data can be n-dimensional.
- The k in KNN stands for the number of datapoints to compare with the new data point. K should be odd. This way a certain category will always win in a vote.
- Downfalls
  - Euclidian distance is used, this might be expensive on huge dataset.
  - Does not perform as well on large datasets compared to other algorithms.
- Can be calculated in parallel. Scales relatively well.
- In the example a dataset from the [UCI machine learning repository][https://archive.ics.uci.edu/ml/datasets.html] is used.

It is very easy and comparable to the linear regression implementation in python:

```python
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True) #missing data is expressed with '?', treat as outlier
df.drop(['id'], 1, inplace=True) #drop the id column, not usable for prediction, makes huge difference

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1]) #random example to classify, you can also give a list of lists to predict multiple cases.
example_measures = example_measures.reshape(1,-1)

prediction = clf.predict(example_measures)

print(prediction)
```

