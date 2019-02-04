import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 

# Data Preparation
df = pd.read_csv('weight-height.csv')
df.head()
df.plot(kind='scatter', x='Height', y='weight', title='Weight and Height in adults')

X = df[['Height']].values
y_true = df['Weight'].values

# Model Building 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.optimizers import Adam, SGD 

model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.summary()
model.compile(Adam(lr=0.8), 'mean_squared_error')
model.fit(X, y_true, epochs=40)

# Model Prediction
y_pred = model.predict(X)

# Plotting 
df.plot(kind='scatter',
        x='Height',
        y='Weight',
        title='Weight and Height in adults')
plt.plot(X, y_pred, color='red')

# Model Evaluation 
W, B = model.get_weights()
W 
B 
from sklearn.metrics import r2_score
print("The R2 score is {:0.3f}".format(r2_score(y_true, y_pred)))

# Model Building using train-test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_true,
                                                    test_size=0.2)
len(X_train)
len(X_test)
W[0, 0] = 0.0
B[0] = 0.0
model.set_weights((W, B))
model.fit(X_train, y_train, epochs=50, verbose=0)
y_train_pred = model.predict(X_train).ravel()
y_test_pred = model.predict(X_test).ravel()

from sklearn.metrics import mean_squared_error as mse
print("The Mean Squared Error on the Train set is:\t{:0.1f}".format(mse(y_train, y_train_pred)))
print("The Mean Squared Error on the Test set is:\t{:0.1f}".format(mse(y_test, y_test_pred)))

print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))
