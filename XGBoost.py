from xgboost import XGBClassifier
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

df = pd.read_csv("train.mv", header=None, sep=' ')
test_df = pd.read_csv("test.mv", header=None, sep=' ')

# taking care of the trailing space at the end of each test instance
test_df.drop(206, axis=1, inplace=True)

# replacing missing values with 0: XGBoost default handler for missing values
df.replace('?', '0', inplace=True)
test_df.replace('?', '0', inplace=True)

for i in range(205):
    # converting the data type of each column to a numeric type
    df[i] = pd.to_numeric(df[i])
    test_df[i] = pd.to_numeric(test_df[i])
    # finding the columns that contain a constant value
    if df[i].unique().size == 1:
        df.drop(i, axis=1, inplace=True)
        test_df.drop(i, axis=1, inplace=True)


# getting rid of class values

# training attributes
X_train = df.drop(205, axis=1).copy()

# testing attributes
X_test = test_df.drop(205, axis=1).copy()


# storing the class values
# training class values
Y_train = df[205].copy()

# X_train = pd.get_dummies(X_train, columns=range(204))
# X_test = pd.get_dummies(X_test, columns=range(204))

# initializing the parameters of the class
# max tree depth = 10
# minimum value (of cover) to accept a leaf = 1
# number of allowed trees = 100
# regularization parameter (lambda) = 1
model = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10, min_child_weight=1, missing=1, n_estimators=100, nthread=-1, objective='binary:logistic', reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=42, silent=True, subsample=1)

model.fit(X_train, Y_train)

# verbose=True,
# early_stopping_rounds=10,
# eval_metric='aucpr',
# eval_set=[(X_train, Y_train)]


# predict the target on the train dataset
predict_train = model.predict(X_train)
# print('\nTarget on train data', predict_train)

# Accuray Score on train dataset
accuracy_train = accuracy_score(Y_train, predict_train)
print('\naccuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = model.predict(X_test)

file = open("class.txt", 'w')

for i in predict_test:
    file.write(str(i) + "\n")

file.close()

# Accuracy Score on test dataset
# accuracy_test = accuracy_score(Y_test, predict_test)
# print('\naccuracy_score on test dataset : ', accuracy_test)


