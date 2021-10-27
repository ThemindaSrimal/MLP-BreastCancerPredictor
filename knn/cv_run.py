import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate


def plot_boxplot(xlabels , yvalues,title):
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    ax.boxplot(yvalues)
    ax.set_xticklabels(xlabels)
    plt.show()


# load data set
path = os.path.dirname(__file__)
DATA_PATH = os.path.join(path, "../data/dataR2.csv")
df = pd.read_csv(DATA_PATH)

# separate data
x = df.drop(columns = ["BMI", "MCP.1", "Age", "Adiponectin", "Leptin", "Classification"], axis=1)  # axis = 1 (specify column)
y = df['Classification']

# use Kfold
#kf = ShuffleSplit(train_size=None,test_size=.3,n_splits=11, random_state=1)
kf= StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
# scoring
scoring = ["accuracy", "f1", "precision", "roc_auc"]
# use svm

classifier = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=19))

# use cross validation

scores = cross_validate(classifier, x, y, scoring=scoring, cv=kf)


# report

yvalues = [scores["test_accuracy"], scores["test_f1"], scores["test_precision"] ,scores["test_roc_auc"]]
xlabels = ["Accuracy", "F1 score", "Precison", "ROC AUC"]
plot_boxplot(xlabels, yvalues,"Results")

new_df = pd.DataFrame(scores, columns=["test_accuracy", "test_f1", "test_precision", "test_roc_auc"])
print(new_df)
