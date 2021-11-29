from xgboost import XGBClassifier
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

# splitting the training data to get validation data
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train,
                                                      test_size=0.1, random_state=None)

# initializing the parameters of the class
# max tree depth = 9
# learning rate = 0.08
# minimum value (of cover) to accept a leaf = 1
# number of allowed trees = 100
# regularization parameter (lambda) = 1
# features size in every subtree = 80%
model = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1,
                          colsample_bytree=0.7, gamma=0, learning_rate=0.08,
                          max_delta_step=0, max_depth=9, min_child_weight=1,
                          missing=1, n_estimators=100, nthread=-1,
                          objective='binary:logistic', reg_alpha=0,
                          reg_lambda=1, scale_pos_weight=1, seed=42, silent=True,
                          subsample=0.8)

# training the model using error in against validation data
model.fit(X_train, Y_train,
          verbose=True,
          early_stopping_rounds=10,
          eval_metric='error',
          eval_set=[(X_valid, Y_valid)])


# predict the target on the train dataset
predict_train = model.predict(X_train)
# print('\nTarget on train data', predict_train)

# Accuracy Score on train dataset
accuracy_train = accuracy_score(Y_train, predict_train)
print('\naccuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = model.predict(X_test)

# writing to the file
file = open("class.txt", 'w')

for i in predict_test:
    file.write(str(i) + "\n")

file.close()

# Accuracy Score on validation dataset
predict_valid = model.predict(X_valid)
accuracy_valid = accuracy_score(Y_valid, predict_valid)
print('\naccuracy_score on validation dataset : ', accuracy_valid)
