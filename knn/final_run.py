import os
import matplotlib.pyplot as plt
import scikitplot as skplt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report



# load data set
path = os.path.dirname(__file__)
DATA_PATH = os.path.join(path, "../data/dataR2.csv")
df = pd.read_csv(DATA_PATH)

# separate data
x = df.drop(columns = ["BMI", "MCP.1", "Age", "Adiponectin", "Leptin", "Classification"], axis=1)  # axis = 1 (specify column)
y = df['Classification']

scaler = StandardScaler().fit(x)
x = scaler.transform(x)


# use Stratified cross validation
kf= StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)

# store splitted data
x_train = []
y_train = []
x_test = []
y_test = []

for train_index, test_index in kf.split(x,y):
    x_train = x[train_index]
    y_train = y[train_index]

    x_test = x[test_index]
    y_test = y[test_index]

# initiate classifier
classifier = KNeighborsClassifier(n_neighbors=19)

# fit data
fit = classifier.fit(x_train, y_train)
y_pred = fit.predict(x_test)
# plot confusion metrix
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)

plt.show()

# plot roc curve
y_probas = fit.predict_proba(x_test)
skplt.metrics.plot_roc(y_test, y_probas)

plt.show()
# plot precision recall curve
skplt.metrics.plot_precision_recall(y_test, y_probas)
plt.show()

# display report
print(classification_report(y_test, y_pred))