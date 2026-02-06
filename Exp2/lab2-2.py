print("Surya Prasad m\n24BAD119")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score,
                              confusion_matrix, roc_curve, auc)

df = pd.read_csv("/home/surya/programs/python-learning/ML lab/LIC.csv")   
df['Price_Movement'] = np.where(df['close'] > df['open'], 1, 0)
features = ['open', 'high', 'low', 'volume']
target = 'Price_Movement'

X = df[features]
y = df[target]
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="AUC = %.2f" % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
importance = log_reg.coef_[0]
plt.figure()
plt.bar(features, importance)
plt.title("Feature Importance")
plt.show()
log_reg_l2 = LogisticRegression(C=0.1, penalty='l2', solver='liblinear')
log_reg_l2.fit(X_train, y_train)

y_pred_tuned = log_reg_l2.predict(X_test)
print("Tuned Accuracy:", accuracy_score(y_test, y_pred_tuned))
