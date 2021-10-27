import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler



# load data set
path = os.path.dirname(__file__)
DATA_PATH = os.path.join(path, "../../data/dataR2.csv")
df = pd.read_csv(DATA_PATH)

# separate data
x = df.drop(columns = 'Classification', axis=1)  # axis = 1 (specify column)
y = df['Classification']

labels = x.columns
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

np.set_printoptions(precision=3)

test = SelectKBest(score_func=mutual_info_classif, k = 9)
fit = test.fit(x, y)

result_df = pd.DataFrame({"Feature":labels, "score":fit.scores_})

print(result_df)

plt.bar(result_df["Feature"], result_df["score"], width=0.4)
plt.title("Feature selection using mutual test")
plt.xticks(rotation=45)
plt.xlabel("Features")
plt.ylabel("score")
plt.show()



