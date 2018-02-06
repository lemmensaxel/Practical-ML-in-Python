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