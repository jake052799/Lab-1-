#Caavs Machine Learning Project Jacob Stys
#Copyright 2022

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset1 = pd.read_csv('Cavs Project FG.csv')
x1 = dataset1.iloc[:, :-1].values
y1 = dataset1.iloc[:, -1].values

print(x1)
print(y1)


dataset2 = pd.read_csv('Cavs Project 3PM.csv')
x2 = dataset2.iloc[:, :-1].values
y2 = dataset2.iloc[:, -1].values

print(x2)
print(y2)


dataset3 = pd.read_csv('Cavs Project FGM and 3PM.csv')
x3 = dataset3.iloc[:, :-1].values
y3 = dataset3.iloc[:, -1].values

print(x3)
print(y3)


# Splitting the 3 Datasets into Training and Test sets

from sklearn.model_selection import train_test_split
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.4, random_state=0)
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.4, random_state=1)
x_train3, x_test3, y_train3, y_test3 = train_test_split(x3, y3, test_size=0.4, random_state=2)

# Feature Scaling on the 3 datasets

from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_train1 = sc1.fit_transform(x_train1)
x_test1 = sc1.transform(x_test1)

sc2 = StandardScaler()
x_train2 = sc2.fit_transform(x_train2)
x_test2 = sc2.transform(x_test2)

sc3 = StandardScaler()
x_train3 = sc3.fit_transform(x_train3)
x_test3 = sc3.transform(x_test3)

print(x_train1)

print(x_train2)

print(x_train3)

# Creating the Decision Tree Model for the three datasets

from sklearn.tree import DecisionTreeClassifier
classifier1 = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier1.fit(x_train1, y_train1)

classifier2 = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier2.fit(x_train2, y_train2)

classifier3 = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier3.fit(x_train3, y_train3)

# Predicting a New Result

print(classifier1.predict(sc1.transform([[40,45]])))

print(classifier2.predict(sc2.transform([[12,45]])))

print(classifier3.predict(sc3.transform([[40,12]])))

# Predicting the Test Set Results

y_pred1 = classifier1.predict(x_test1)
print(np.concatenate((y_pred1.reshape(len(y_pred1), 1), y_test1.reshape(len(y_test1),1)),1))

y_pred2 = classifier1.predict(x_test2)
print(np.concatenate((y_pred2.reshape(len(y_pred2), 1), y_test2.reshape(len(y_test2),1)),1))

y_pred3 = classifier3.predict(x_test3)
print(np.concatenate((y_pred3.reshape(len(y_pred3), 1), y_test3.reshape(len(y_test3),1)),1))

# Making the Confusion Matrix and Accuracy Score

from sklearn.metrics import confusion_matrix, accuracy_score
cm1 = confusion_matrix(y_test1, y_pred1)
print(cm1)
ac1 = accuracy_score(y_test1, y_pred1)
print(ac1)

cm2 = confusion_matrix(y_test2, y_pred2)
print(cm2)
ac2 = accuracy_score(y_test2, y_pred2)
print(ac2)

cm3 = confusion_matrix(y_test3, y_pred3)
print(cm3)
ac3 = accuracy_score(y_test3, y_pred3)
print(ac3)

#Visualising the Training Results for Dataset 1

from matplotlib.colors import ListedColormap
x_set1, y_set1 = sc1.inverse_transform(x_train1), y_train1
x1, x2 = np.meshgrid(np.arange(start= x_set1[:,0].min() - 10, stop=x_set1[:, 0].max() + 10, step = .25),
                     np.arange(start= x_set1[:, 1].min() - 10, stop= x_set1[:, 1].max()+10, step=.25))
plt.contourf(x1, x2, classifier1.predict(sc1.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
             alpha=0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set1)):
    plt.scatter(x_set1[y_set1 == j, 0], x_set1[y_set1 == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Decision Tree Classification with FGM and Rebounds (Training Set)')
    plt.xlabel('FGM')
    plt.ylabel('Rebounds')
    plt.legend()
    plt.show()

#Visualising the Training Results for Dataset 1

from matplotlib.colors import ListedColormap
x_set1, y_set1 = sc1.inverse_transform(x_test1), y_test1
x1, x2 = np.meshgrid(np.arange(start= x_set1[:,0].min() - 10, stop=x_set1[:, 0].max() + 10, step = .25),
                     np.arange(start= x_set1[:, 1].min() - 10, stop= x_set1[:, 1].max()+10, step=.25))
plt.contourf(x1, x2, classifier1.predict(sc1.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
             alpha=0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set1)):
    plt.scatter(x_set1[y_set1 == j, 0], x_set1[y_set1 == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Decision Tree Classification with FGM and Rebounds (Test Set)')
    plt.xlabel('FGM')
    plt.ylabel('Rebounds')
    plt.legend()
    plt.show()

#Visualising the Training Results for Dataset 1

from matplotlib.colors import ListedColormap
x_set2, y_set2 = sc2.inverse_transform(x_train2), y_train2
x3, x4 = np.meshgrid(np.arange(start= x_set2[:,0].min() - 10, stop=x_set2[:, 0].max() + 10, step = .25),
                     np.arange(start= x_set2[:, 1].min() - 10, stop= x_set2[:, 1].max()+10, step=.25))
plt.contourf(x3, x4, classifier2.predict(sc2.transform(np.array([x3.ravel(), x4.ravel()]).T)).reshape(x3.shape),
             alpha=0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x3.min(), x3.max())
plt.ylim(x4.min(), x4.max())
for i, j in enumerate(np.unique(y_set2)):
    plt.scatter(x_set2[y_set2 == j, 0], x_set2[y_set2 == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Decision Tree Classification with 3PM and Rebounds (Training Set)')
    plt.xlabel('3PM')
    plt.ylabel('Rebounds')
    plt.legend()
    plt.show()

#Visualising the Training Results for Dataset 1

from matplotlib.colors import ListedColormap
x_set2, y_set2 = sc2.inverse_transform(x_test2), y_test2
x3, x4 = np.meshgrid(np.arange(start= x_set2[:,0].min() - 10, stop=x_set2[:, 0].max() + 10, step = .25),
                     np.arange(start= x_set2[:, 1].min() - 10, stop= x_set2[:, 1].max()+10, step=.25))
plt.contourf(x3, x4, classifier2.predict(sc2.transform(np.array([x3.ravel(), x4.ravel()]).T)).reshape(x3.shape),
             alpha=0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x3.min(), x3.max())
plt.ylim(x4.min(), x4.max())
for i, j in enumerate(np.unique(y_set2)):
    plt.scatter(x_set2[y_set2 == j, 0], x_set2[y_set2 == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Decision Tree Classification with 3PM and Rebounds (Test Set)')
    plt.xlabel('3PM')
    plt.ylabel('Rebounds')
    plt.legend()
    plt.show()