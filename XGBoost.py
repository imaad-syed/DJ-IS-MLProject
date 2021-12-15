from xgboost import XGBClassifier
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

df = pd.read_csv("train.mv", header=None, sep=' ')
test_df = pd.read_csv("test.mv", header=None, sep=' ')
#old_test_df = pd.read_csv("old_test.mv", header=None, sep=' ')

# taking care of the trailing space at the end of each test instance
#test_df.drop(206, axis=1, inplace=True)  #.. the space has been removed in the new file
#old_test_df.drop(206, axis=1, inplace=True)

# replacing missing values with 0: XGBoost default handler for missing values
df.replace('?', '0', inplace=True)
test_df.replace('?', '0', inplace=True)
#old_test_df.replace('?', '0', inplace=True)


for i in range(205):
    # converting the data type of each column to a numeric type
    df[i] = pd.to_numeric(df[i])
    test_df[i] = pd.to_numeric(test_df[i])
    # old_test_df[i] = pd.to_numeric(old_test_df[i])
    # finding the columns that contain a constant value
    if df[i].unique().size == 1:
        df.drop(i, axis=1, inplace=True)
        test_df.drop(i, axis=1, inplace=True)
        # old_test_df.drop(i, axis=1, inplace=True)


# getting rid of class values

# training attributes
X_train = df.drop(205, axis=1).copy()

# testing attributes
X_test = test_df.drop(205, axis=1).copy()

# storing the class values
# training class values
Y_train = df[205].copy()



# initializing the parameters of the class
# max tree depth = 9
# learning rate = 0.07
# minimum value (of cover) to accept a leaf = 1
# number of allowed trees = 152
# regularization parameter (lambda) = 1
# features size in every subtree = 78%
model = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1,
                          colsample_bytree=0.7, gamma=0, learning_rate=0.07,
                          max_delta_step=0, max_depth=9, min_child_weight=1,
                          missing=1, n_estimators=152, nthread=-1,
                          objective='binary:logistic', reg_alpha=0.25,
                          reg_lambda=1, scale_pos_weight=1, seed=42, verbosity=0, silent=True,
                          subsample=0.78, use_label_encoder=False)

# k =20
kfold = KFold(n_splits=20)

# training the model using error in against validation data
model.fit(X_train, Y_train)


# predict the target on the train dataset
# predict_train = model.predict(X_train)

# Accuracy Score on train dataset
accuracy_valid = cross_val_score(model, X_train, Y_train, cv=kfold)

# accuracy_train = accuracy_score(Y_train, predict_train)
print('\naccuracy_score on validation datasets : ', accuracy_valid*100, '\nmean accuracy_score on validation datasets : ', accuracy_valid.mean()*100, '\nstandard deviation accuracy_score on validation datasets : ',accuracy_valid.std()*100)

# predict the target on the test dataset
predict_valid_test = model.predict(X_test)

# writing to the file 1
file = open("output1.txt", 'w')

count = 0
for i in predict_valid_test:
    count = count + 1
    if count != 4000:
        file.write(str(i) + "\n")
    else:
        file.write(str(i))

file.close()


######

# training to maximize accuracy on preliminary test dataset (using hyper parameters tuned to maximize that accuracy)

#####

# training instances
X_train = df.drop(205, axis=1).copy()

# training class values
Y_train = df[205].copy()

# testing instances
# X_test = old_test_df.drop(205, axis=1).copy()

# testing class values
# Y_test = old_test_df[205].copy()
# Y_test = pd.to_numeric(Y_test)

model2 = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1,
                           colsample_bytree=0.7, gamma=0, learning_rate=0.07,
                           max_delta_step=0, max_depth=9, min_child_weight=1,
                           missing=1, n_estimators=152, nthread=-1,
                           objective='binary:logistic', reg_alpha=0,
                           reg_lambda=1, scale_pos_weight=1, seed=42, silent=True,
                           subsample=0.78)

model2.fit(X_train, Y_train)

# predict the target on the test dataset
predict_test = model2.predict(X_test)

# Accuracy Score on test dataset

# accuracy_test = accuracy_score(Y_test, predict_test)
# print('\naccuracy_score on preliminary test dataset : ', accuracy_test*100)

# writing to the file 2
file = open("output2.txt", 'w')

count = 0
for i in predict_test:
    count = count + 1
    if count != 4000:
        file.write(str(i) + "\n")
    else:
        file.write(str(i))

file.close()
