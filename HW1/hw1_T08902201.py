import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def oneHotEncoded(data_frame, column_name_ls):
    df = data_frame.copy(deep=True)
    for col in column_name_ls:
        df[col] = pd.Categorical(df[col])
        df_dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, df_dummies], axis=1)
        df.drop([col], axis=1, inplace=True)

    return df


def normalize(data_frame):
    df = data_frame.copy(deep=True)

    return (df - df.mean()) / df.std()


def compute_RMSE(y, y_predict):
    sigma = np.square(y_predict - y).sum()
    return np.sqrt(sigma / y.shape[0])


def pseudo_inverse(x, y):
    inverse = np.linalg.pinv(np.dot(x.T, x))
    temp = np.dot(x.T, y)
    return np.dot(inverse, temp)


def pseudo_inverse_reg(x, y):
    inverse = np.linalg.pinv(np.dot(x.T, x) + 0.5 * np.identity(x.shape[1]))
    temp = np.dot(x.T, y)
    return np.dot(inverse, temp)


def pseudo_inverse_reg_bias(x, y):
    iden = np.identity(x.shape[1])
    iden[0][0] = 0
    inverse = np.linalg.pinv(np.dot(x.T, x) + 0.5 * iden)
    temp = np.dot(x.T, y)
    return np.dot(inverse, temp)


data_all = pd.read_csv("train.csv")
y = data_all[['G3']]
train_set = data_all[['school', 'sex', 'age', 'famsize', 'studytime', 'failures', 'activities', 'higher',
                      'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']]

column_needed_onehot = ['school', 'sex', 'famsize', 'activities',
                        'higher', 'internet', 'romantic']

train_set = oneHotEncoded(train_set, column_needed_onehot)

x_train, x_test, y_train, y_test = train_test_split(train_set, y, test_size=0.2)

mean = x_train.mean()
sd = x_train.std()
x_train = (x_train - mean) / sd
x_test = (x_test - mean) / sd

# pseudo-inverse without bias
weight = pseudo_inverse(x_train.values, y_train.values)

# Find root mean square
y_predict = np.dot(x_test.values, weight)
RMSE = compute_RMSE(y_test.values, y_predict)
print(RMSE)
print('Linear regression RMSE = ' + str(RMSE))

# pseudo-inverse without bias + regularization
weight_reg = pseudo_inverse_reg(x_train.values, y_train.values)

y_predict_reg = np.dot(x_test.values, weight_reg)
RMSE_reg = compute_RMSE(y_test.values, y_predict_reg)
print('Linear regression(reg) RMSE = ' + str(RMSE_reg))

# add bias to x_train, x_test
x_train_bias = x_train.copy(deep=True)
x_train_bias.insert(0, column='bias', value=np.ones((x_train_bias.shape[0], 1)))

x_test_bias = x_test.copy(deep=True)
x_test_bias.insert(0, column='bias', value=np.ones((x_test_bias.shape[0], 1)))

# pseudo-inverse without bias + regularization + bias
weight_reg_bias = pseudo_inverse_reg_bias(x_train_bias.values, y_train.values)

y_predict_reg_bias = np.dot(x_test_bias.values, weight_reg_bias)
RMSE_reg_bias = compute_RMSE(y_test.values, y_predict_reg_bias)
print('Linear regression(r/b) RMSE = ' + str(RMSE_reg_bias))

# Bayesian Linear Regression
reg = 1 * np.identity(x_train_bias.shape[1])
reg[0][0] = 0

x = x_train_bias.values

inverse = np.linalg.pinv(np.dot(x.T, x) + reg)
temp = np.dot(x.T, y_train.values)
weight_bayesian = np.dot(inverse, temp)

y_predict_bayesian = np.dot(x_test_bias.values, weight_bayesian)
RMSE_bayesian = compute_RMSE(y_test.values, y_predict_bayesian)
print('Bayesian Linear regression RMSE = ' + str(RMSE_bayesian))

X = range(0, y_test.shape[0])

# plt.figure(figsize=(, 5))
plt.xlabel("Sample Index")
plt.ylabel("Values")

plt.plot(X, y_test, 'blue', label="Ground Truth")
plt.plot(X, y_predict, 'orange', label="Linear Regression " + str(round(RMSE, 5)))
plt.plot(X, y_predict_reg, 'green', label="Linear Regression (Reg) " + str(round(RMSE_reg, 5)))
plt.plot(X, y_predict_reg_bias, 'purple', label="Linear Regression (r/b) " + str(round(RMSE_reg_bias, 5)))
plt.plot(X, y_predict_bayesian, 'orange', label="Bayesian Liinear Regression" + str(round(RMSE_bayesian, 5)))

plt.legend()
plt.show()


test_set_all = pd.read_csv('test_no_G3.csv')
ids = test_set_all['ID']
test_set = test_set_all[['school', 'sex', 'age', 'famsize', 'studytime', 'failures', 'activities', 'higher',
                         'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']]

test_set = oneHotEncoded(test_set, column_needed_onehot)
test_set_norm = normalize(test_set)


test_set_norm.insert(0, column='bias', value=np.ones((test_set_norm.shape[0], 1)))

y_predict_test = np.dot(test_set_norm.values, weight_reg_bias)

with open('T08902201_1.txt', 'w') as f:
    i = 0
    for y_predicted in y_predict_test:
        row = str(ids[i]) + '\t' + str(y_predicted[0]) + '\n'
        i = i + 1
        f.write(row)
