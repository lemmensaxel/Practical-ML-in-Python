import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

df = quandl.get('WIKI/GOOGL') #df = dataframe

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']] #Select attributes from dataset
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0 #Adding combined attributes
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 #Adding combined attributes

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']] # Selecte the actual dataset to work with

df.fillna(-99999, inplace=True) #Replace NaN data with -99999 inplace, to treat is as an outlier in the data.

forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.01*len(df))) #We try to predict out of 1% of the dataframe.

df['label'] = df[forecast_col].shift(-forecast_out) #Each row of the label column will be the "Adj. Close" price "forecast_out"-days into the future

x = np.array(df.drop(['label'], 1)) #convert df into new dataframe with only features and place it in numpy array.
x = preprocessing.scale(x) #rescale/normalize data
x_lately = x[-forecast_out:] #We don't have label-values associated with these x-values
x = x[:-forecast_out:]


df.dropna(inplace=True)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2) # Take all labels and features, shuffle them up while keeping xes and ys paired. It outputs x and y training data and testing data.
clf = LinearRegression() #create classifier
clf.fit(x_train, y_train) #Train the classifier using generated training data

#Let's save the trained classifier
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

#To use it again
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

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
