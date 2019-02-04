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
from keras.wrappers.scikit_learn import KerasClassifier

def build_logistic_regression_model():
    model = Sequential()
    model.add(Dense(1, input_shape=(1,), activation='sigmoid'))
    model.compile(SGD(lr=0.5),
                  'binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=build_logistic_regression_model,
                        epochs=25,
                        verbose=0)

from sklearn.model_selection import cross_val_score, KFold
cv = KFold(3, shuffle=True)
scores = cross_val_score(model, X, y, cv=cv)
scores 

# Evaluation
print("The cross validation accuracy is {:0.4f} ± {:0.4f}".format(scores.mean(), scores.std()))
