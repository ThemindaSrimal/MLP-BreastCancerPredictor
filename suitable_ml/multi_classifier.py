import os
import  matplotlib.pyplot as plt
from matplotlib.pyplot import plot_date
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold ,StratifiedShuffleSplit, cross_val_score, cross_validate

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

# define kfold
K = 10

# separate data
# x = df.drop(columns = 'Classification', axis=1)  # axis = 1 (specify column)
x = df.drop(columns = ["BMI", "MCP.1", "Age", "Adiponectin", "Leptin", "Classification"], axis=1)  # axis = 1 (specify column)
y = df['Classification']

# standardize data
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

models = []
models.append(("SVC" , SVC(kernel="sigmoid", C=0.6,gamma="auto")))
models.append(("Logistic Regression", LogisticRegression(solver="liblinear", C=0.3)))
models.append(("Random forest", RandomForestClassifier(n_estimators=5)))
models.append(("KNeighbour", KNeighborsClassifier(n_neighbors=19)))

results = []
names = []
total_df = []

acc_results = []
f1_results = []
precision_results = []

for name, model in models:

    # use Kfold
    kf = KFold(n_splits=K)
    kf2 = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    # use cross validation
    score = cross_val_score(model, x, y, scoring="accuracy", cv=kf)

    results.append(score)
    names.append(name)
    temp = cross_validate(model, x, y, scoring=["accuracy","f1","precision"], cv=kf2)

    acc_results.append(temp["test_accuracy"])
    f1_results.append(temp["test_f1"])
    precision_results.append(temp["test_precision"])

    total_df.append(
            {
                "name":name,
                "test_accuracy" : temp["test_accuracy"].mean(),
                "test_f1": temp["test_f1"].mean(),
                "test_precision" : temp["test_precision"].mean(),
            }
    )

# report

plot_boxplot(names, results,"Compare models test scores")
plot_boxplot(names,acc_results, "Compare models test accuracy")
plot_boxplot(names,f1_results, "Compare models test f1 score")
plot_boxplot(names,precision_results, "Compare models test precision")

new_df = pd.DataFrame(total_df, columns=["name", "test_accuracy","test_f1","test_precision"])
print(new_df)