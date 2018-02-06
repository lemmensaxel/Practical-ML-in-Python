## Regression

- The whole point of regession is to fit a function to a dataset. 


- It is commonly used in stock pricing.
- [Quandle][https://www.quandl.com/] is a source of financial, economic and alternative data. Without an account, restricted to few request/day.

Everything boils down to features and labels.

- **Features**: Categories of metrics. Sometimes a lot of correlation between the different features. These are the attributes that make up/cause the label values.
- **Labels**: Predictions into the future.

```python
import pandas as pd
import quandl

df = quandl.get('WIKI/GOOGL') #df = dataframe

#print(df.head())

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']] #Select attributes from dataset
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0 #Adding combined attributes
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 #Adding combined attributes

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']] # Selecte the actual dataset to work with
```

Is `Adj. Close` a feature or a label? It is a feature (if for example the last 10 values are grouped) or non of both.

We can then add our labels to the code:

```python
df.fillna(-99999, inplace=True) #Replace NaN data with -99999 inplace, to treat is as an outlier in the data.

forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.01*len(df))) #We try to predict out of 1% of the dataframe.

df['label'] = df[forecast_col].shift(-forecast_out) #Each row of the label column will be the "Adj. Close" price "forecast_out"-days into the future

```

Now we use our dataframe to perform some actual classifer training:

```Python
import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL') #df = dataframe

#print(df.head())

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']] #Select attributes from dataset
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0 #Adding combined attributes
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 #Adding combined attributes

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']] # Selecte the actual dataset to work with

df.fillna(-99999, inplace=True) #Replace NaN data with -99999 inplace, to treat is as an outlier in the data.

forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.01*len(df))) #We try to predict out of 1% of the dataframe.

df['label'] = df[forecast_col].shift(-forecast_out) #Each row of the label column will be the "Adj. Close" price "forecast_out"-days into the future
df.dropna(inplace=True)

x = np.array(df.drop(['label'], 1)) #convert df into new dataframe with only features and place it in numpy array.
x = preprocessing.scale(x) #rescale/normalize data
y = np.array(df['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2) # Take all labels and features, shuffle them up while keeping xes and ys paired. It outputs x and y training data and testing data.
clf = LinearRegression() #create classifier
clf.fit(x_train, y_train) #Train the classifier using generated training data
accuracy = clf.score(x_test, y_test) #Test the classifier

print(forecast_out)
print(accuracy) # We can predict with this accuracy, forecast_out days in advance using our modelled classifier. This is R-squared accuracy.
```

Always check the docs of your used algorithm (linear regression in this case) to check if it can be threaded:

```python
clf = LinearRegression(n_jobs=10) #create classifier to run over 10 threads
clf = LinearRegression(n_jobs=-1) #create classifier to run over as many threads as possible
```

We can now use this technique to predict future values:

``` python
x = np.array(df.drop(['label'], 1)) #convert df into new dataframe with only features and place it in numpy array.
x = preprocessing.scale(x) #rescale/normalize data
x_lately = x[-forecast_out:] #We don't have label-values associated with these x-values
x = x[:-forecast_out:]


df.dropna(inplace=True)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2) # Take all labels and features, shuffle them up while keeping xes and ys paired. It outputs x and y training data and testing data.
clf = LinearRegression() #create classifier
clf.fit(x_train, y_train) #Train the classifier using generated training data
accuracy = clf.score(x_test, y_test) #Test the classifier

#print(accuracy)

#let's predict some stuff

forecast_set = clf.predict(x_lately) #Predict the label values associated with the latest features

style.use('ggplot')
df['Forecast'] = np.nan #Fill with nan data

last_data = df.iloc[-1].name
last_unix = last_data.timestamp()
one_day = 86400
next_unix = last_unix + one_day

#Some crap to put dates on the x-axis
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
```

### Extra: Pickling and Scaling

- **Pickeling**: Serialization of any python object. In this case a classifier, to avoid having to train it again.

```Python
clf.fit(x_train, y_train) #Train the classifier using generated training data

#Let's save the trained classifier
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

#To use it again
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)
```

This comes in extremely handy when dealing with huge classifiers. You can just rent some powerful server to train a classifier and save it as a pickle.
