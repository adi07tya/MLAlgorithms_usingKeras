import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 

# Data Preparation
df = pd.read_csv('user_visit_duration.csv')
df.head()
df.plot(kind='scatter', x='Time (min)', y='Buy')

X = df[['Time (min)']].values
y = df['Buy'].values

# Model Building 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.optimizers import Adam, SGD 

model = Sequential()
model.add(Dense(1, input_shape=(1,), activation='sigmoid'))
model.summary()
model.compile(SGD(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=25)

# Model Prediction
y_pred = model.predict(X)
y_class_pred = y_pred > 0.5

# Plotting
ax = df.plot(kind='scatter', x='Time (min)', y='Buy',
             title='Purchase behavior VS time spent on site')

temp = np.linspace(0, 4)
ax.plot(temp, model.predict(temp), color='orange')
plt.legend(['model', 'data'])

# Model Evaluation 
from sklearn.metrics import accuracy_score
print("The accuracy score is {:0.3f}".format(accuracy_score(y, y_class_pred)))

# Model Building using train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Setting the parameters to zero here 
params = model.get_weights()
params = [np.zeros(w.shape) for w in params]
model.set_weights(params)

print("The accuracy score is {:0.3f}".format(accuracy_score(y, model.predict(X) > 0.5)))
model.fit(X_train, y_train, epochs=25, verbose=0)
print("The train accuracy score is {:0.3f}".format(accuracy_score(y_train, model.predict(X_train) > 0.5)))
print("The test accuracy score is {:0.3f}".format(accuracy_score(y_test, model.predict(X_test) > 0.5)))
