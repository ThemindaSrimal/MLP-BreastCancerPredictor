import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold , cross_val_score, GridSearchCV

# load data set
path = os.path.dirname(__file__)
DATA_PATH = os.path.join(path, "../data/dataR2.csv")
df = pd.read_csv(DATA_PATH)

# define kfold
K = 10

# separate data
x = df.drop(columns = ["BMI", "MCP.1", "Age", "Adiponectin", "Leptin", "Classification"], axis=1)  # axis = 1 (specify column)
y = df['Classification']

# standardize data
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

models = {
    "svm" : {
        "model": SVC(gamma="auto"),
        "params": {
            "C": [0.1 , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 10 , 20, 30 ],
            "kernel": ("rbf", "linear","sigmoid", "poly")
        }
    },
    "logistic_regression" : {
         "model": LogisticRegression(solver="liblinear"),
        "params":{
            "C":  [0.1 , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 10 , 20, 30 ]
        }
    },
    "random_forest" : {
         "model":RandomForestClassifier(),
        "params":{
            "n_estimators": [1, 5, 10]
        }
    },
    "kNeighbor": {
        "model":KNeighborsClassifier(),
        "params":{
            "n_neighbors": range(1,100),
            "weights": ["uniform", "distance"]
        }
    }
}


results = []
names = []

for name, model_param in models.items():

    # use Kfold
    kf =KFold(n_splits=K)
    # use cross validation
    search = GridSearchCV(estimator=model_param["model"], param_grid=model_param["params"], cv= kf,n_jobs=-1)
    search.fit(x, y)
    results.append({
        "model": name,
        "best_score": search.best_score_,
        "best_params": search.best_params_
    })


#report

result_df = pd.DataFrame(results, columns=["model", "best_score", "best_params"])

print(result_df)
plt.bar(result_df["model"],result_df["best_score"], width=0.4)

plt.show()

