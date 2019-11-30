import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, f1_score, roc_curve, \
    auc, precision_recall_curve, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from inspect import signature

import seaborn as sns

data_all = pd.read_csv('Data.csv')

data_all['normalizedAmount'] = StandardScaler().fit_transform(data_all['Amount'].values.reshape(-1, 1))
# Drop 'Time' feature because I think this feature is not related to the prediction
data_all.drop(['Amount', 'Time'], axis=1, inplace=True)

train = data_all.drop(['Class'], axis=1)
label = data_all['Class']
train_set, test_set, train_label, test_label = train_test_split(train, label, test_size=0.2)

#  <-------------DNN----------------->

model = keras.Sequential([
    keras.layers.Dense(29, activation='relu', input_shape=(29,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
adam = keras.optimizers.Adam(learning_rate=0.003)
model.compile(optimizer=adam, loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_set, train_label, batch_size=64, epochs=20, validation_split=0.2)

train_loss, train_acc = model.evaluate(train_set, train_label)
print('DNN trained Acc: ', train_acc)
print('DNN trained Loss: ', train_loss)

test_loss, test_acc = model.evaluate(test_set, test_label)
print('DNN tested Acc: ', test_acc)
print('DNN tested Loss: ', test_loss)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# train confusion matrix
train_pred = np.round(model.predict(train_set))
cm_train = confusion_matrix(train_label, train_pred)

ax = plt.subplot(2, 2, 1)  #####
matrix = sns.heatmap(cm_train, annot=True, ax=ax, fmt='d',
                     xticklabels=['predict=0', 'predict=1'],
                     yticklabels=['True=0', 'True=1'])
bottom, top = matrix.get_ylim()
matrix.set_ylim(bottom + 0.5, top - 0.5)

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Train Confusion Matrix')

# validation confusion matrix
test_pred = np.round(model.predict(test_set))
cm_test = confusion_matrix(test_label, test_pred)

ax2 = plt.subplot(2, 2, 2)  #####
matrix2 = sns.heatmap(cm_test, annot=True, ax=ax2, fmt='d',
                      xticklabels=['predict=0', 'predict=1'],
                      yticklabels=['True=0', 'True=1'])  # annot=True to annotate cells
bottom, top = matrix2.get_ylim()
matrix2.set_ylim(bottom + 0.5, top - 0.5)

ax2.set_xlabel('Predicted labels')
ax2.set_ylabel('True labels')
ax2.set_title('Validation Confusion Matrix')

precision, recall, fscore, _ = precision_recall_fscore_support(test_label, test_pred)
print('\033[1mClass 0\033[0m')
print('Precision: ', precision[0])
print('Recall: ', recall[0])
print('F-score: ', fscore[0])
print('\033[1mClass 1\033[0m')
print('Precision: ', precision[1])
print('Recall: ', recall[1])
print('F-score: ', fscore[1])
print('\033[1mAverage\033[0m')
print('Precision: ', np.average(precision))
print('Recall: ', np.average(recall))
print('F-score: ', np.average(fscore))

# ROC curve plot
fpr, tpr, _ = roc_curve(test_label, test_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Precision recall curve
precision, recall, _ = precision_recall_curve(test_label, test_pred)

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
    np.average(precision)))
plt.show()

# <-------------Hidden Data----------------->
hidden_data = pd.read_csv('test_no_Class.csv')
hidden_data['normalizedAmount'] = StandardScaler().fit_transform(hidden_data['Amount'].values.reshape(-1, 1))
hidden_data.drop(['Amount', 'Time'], axis=1, inplace=True)
hidden_pred = np.round(model.predict(hidden_data))

with open('T08902201_answer.txt', 'w') as f:
    i = 0
    for pred in hidden_pred:
        row = str(int(pred[0])) + '\n'
        i = i + 1
        f.write(row)

# <-------------Decision Tree----------------->

tree = DecisionTreeClassifier()
tree = tree.fit(train_set, train_label)
tree_pred = tree.predict(test_set)

# Confusion matrix
tree_cm = confusion_matrix(test_label, tree_pred)

tp = tree_cm[0, 0]
tn = tree_cm[1, 1]
fp = tree_cm[1, 0]
fn = tree_cm[0, 1]

p = tp / (tp + fp)
r = tp / (tp + fn)
print('\033[1mDecision Tree\033[0m')
print("Accuracy:", accuracy_score(test_label, tree_pred))
print('Precision:', p)
print('Recall:', r)
print('F1_score:', (2 * p * r) / (p + r))

ax3 = plt.subplot(2, 2, 3)  ######
matrix3 = sns.heatmap(tree_cm, annot=True, ax=ax3, fmt='d',
                      xticklabels=['predict=0', 'predict=1'],
                      yticklabels=['True=1', 'True=0'])  # annot=True to annotate cells
bottom, top = matrix3.get_ylim()
matrix3.set_ylim(bottom + 0.5, top - 0.5)

ax3.set_xlabel('Predicted labels')
ax3.set_ylabel('True labels')
ax3.set_title('Tree Confusion Matrix')

# <-------------Random Forest----------------->

rf = RandomForestClassifier(n_estimators=100)
rf.fit(train_set, train_label)
rf_pred = rf.predict(test_set)
feature_imp = pd.Series(rf.feature_importances_, index=train_set.columns).sort_values(ascending=False)

# Confusion matrix
rf_cm = confusion_matrix(test_label, rf_pred)

tp = rf_cm[0, 0]
tn = rf_cm[1, 1]
fp = rf_cm[1, 0]
fn = rf_cm[0, 1]

p = tp / (tp + fp)
r = tp / (tp + fn)
print('\033[1mRandom Forest\033[0m')
print("Accuracy:", accuracy_score(test_label, rf_pred))
print('Precision:', p)
print('Recall:', r)
print('F1_score:', (2 * p * r) / (p + r))

ax4 = plt.subplot(2, 2, 4)  ######
matrix4 = sns.heatmap(rf_cm, annot=True, ax=ax4, fmt='d',
                      xticklabels=['predict=0', 'predict=1'],
                      yticklabels=['True=1', 'True=0'])
bottom, top = matrix4.get_ylim()
matrix4.set_ylim(bottom + 0.5, top - 0.5)

ax4.set_xlabel('Predicted labels')
ax4.set_ylabel('True labels')
ax4.set_title('Random Forest Confusion Matrix')
plt.show()
