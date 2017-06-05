import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("iris.csv")

df = df[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species"]]

X = np.array(df.drop(["Species"],1))
y = np.array(df["Species"])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = LogisticRegression()
clf.fit(X_train, y_train)
conf = clf.score(X_test, y_test)
print(conf)
