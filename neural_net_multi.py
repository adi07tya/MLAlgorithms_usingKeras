import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

# Data Preparation
df = pd.read_csv('iris.csv')

import seaborn as sns
sns.pairplot(df, hue="species")
df.head()

X = df.drop('species', axis=1)
X.head()

target_names = df['species'].unique()
target_names

target_dict = {n:i for i, n in enumerate(target_names)}
target_dict

y= df['species'].map(target_dict)
y.head()
from keras.utils.np_utils import to_categorical
y_cat = to_categorical(y)
y_cat[:10]
X_train, X_test, y_train, y_test = train_test_split(X.values, y_cat,
                                                    test_size=0.2)

# Model Building 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

model = Sequential()
model.add(Dense(3, input_shape=(4,), activation='softmax'))
model.compile(Adam(lr=0.1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, validation_split=0.1)
y_pred = model.predict(X_test)
y_pred[:5]

y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report
print(classification_report(y_test_class, y_pred_class))
confusion_matrix(y_test_class, y_pred_class)
