 # Data Exploration with Pandas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv('titanic-train.csv')

type(df)
df.head()
df.info()
df.describe()


# Indexing
df.iloc[3]
df.loc[0:4,'Ticket']
df['Ticket'].head()
df[['Embarked', 'Ticket']].head()


# Selections
df[df['Age'] > 70]
df['Age'] > 70
df.query("Age > 70")
df[(df['Age'] == 11) & (df['SibSp'] == 5)]
df[(df.Age == 11) | (df.SibSp == 5)]
df.query('(Age == 11) | (SibSp == 5)')


# Unique Values
df['Embarked'].unique()


# Sorting
df.sort_values('Age', ascending = False).head()


# Aggregations
df['Survived'].value_counts()
df['Pclass'].value_counts()
df.groupby(['Pclass', 'Survived'])['PassengerId'].count()
df['Age'].min()
df['Age'].max()
df['Age'].mean()
df['Age'].median()

mean_age_by_survived = df.groupby('Survived')['Age'].mean()
mean_age_by_survived

std_age_by_survived = df.groupby('Survived')['Age'].std()
std_age_by_survived


# Merge
df1 = mean_age_by_survived.round(0).reset_index()
df2 = std_age_by_survived.round(0).reset_index()
df1
df2
df3 = pd.merge(df1, df2, on='Survived')
df3
df3.columns = ['Survived', 'Average Age', 'Age Standard Deviation']
df3


# Pivot Tables
df.pivot_table(index='Pclass',
               columns='Survived',
               values='PassengerId',
               aggfunc='count')


# Correlations
df['IsFemale'] = df['Sex'] == 'female'
correlated_with_survived = df.corr()['Survived'].sort_values()
correlated_with_survived
correlated_with_survived.iloc[:-1].plot(kind='bar',
                                        title='Titanic Passengers: correlation with survival')


# Visual Data Exploration with Matplotlib
data1 = np.random.normal(0, 0.1, 1000)
data2 = np.random.normal(1, 0.4, 1000) + np.linspace(0, 1, 1000)
data3 = 2 + np.random.random(1000) * np.linspace(1, 5, 1000)
data4 = np.random.normal(3, 0.2, 1000) + 0.3 * np.sin(np.linspace(0, 20, 1000))

data = np.vstack([data1, data2, data3, data4]).transpose()

df = pd.DataFrame(data, columns=['data1', 'data2', 'data3', 'data4'])
df.head()


# Line Plot
df.plot(title='Line plot')
plt.plot(df)
plt.title('Line plot')
plt.legend(['data1', 'data2', 'data3', 'data4'])


# Scatter Plot
df.plot(style='.')
_ = df.plot(kind='scatter', x='data1', y='data2',
            xlim=(-1.5, 1.5), ylim=(0, 3))


# Histograms
df.plot(kind='hist',
        bins=50,
        title='Histogram',
        alpha=0.6)


# Cumulative distribution
df.plot(kind='hist',
        bins=100,
        title='Cumulative distributions',
        normed=True,
        cumulative=True,
        alpha=0.4)


# Box Plot
df.plot(kind='box',
        title='Boxplot')


# Subplots
fig, ax = plt.subplots(2, 2, figsize=(5, 5))

df.plot(ax=ax[0][0],
        title='Line plot')

df.plot(ax=ax[0][1],
        style='o',
        title='Scatter plot')

df.plot(ax=ax[1][0],
        kind='hist',
        bins=50,
        title='Histogram')

df.plot(ax=ax[1][1],
        kind='box',
        title='Boxplot')

plt.tight_layout()


# Pie charts
gt01 = df['data1'] > 0.1
piecounts = gt01.value_counts()
piecounts
piecounts.plot(kind='pie',
               figsize=(5, 5),
               explode=[0, 0.15],
               labels=['<= 0.1', '> 0.1'],
               autopct='%1.1f%%',
               shadow=True,
               startangle=90,
               fontsize=16)


# Hexbin plot
data = np.vstack([np.random.normal((0, 0), 2, size=(1000, 2)),
                  np.random.normal((9, 9), 3, size=(2000, 2))])
df = pd.DataFrame(data, columns=['x', 'y'])
df.head()
df.plot()
df.plot(kind='kde')
df.plot(kind='hexbin', x='x', y='y', bins=100, cmap='rainbow')


# Unstructured data

# Images
from PIL import Image
img = Image.open('iss.jpg')
img
type(img)
imgarray = np.asarray(img)
type(imgarray)
imgarray.shape
imgarray.ravel().shape
435 * 640 * 3


# Sound
from scipy.io import wavfile
rate, snd = wavfile.read(filename='sms.wav')
from IPython.display import Audio
Audio(data=snd, rate=rate)
len(snd)
snd
plt.plot(snd)
_ = plt.specgram(snd, NFFT=1024, Fs=44100)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')


