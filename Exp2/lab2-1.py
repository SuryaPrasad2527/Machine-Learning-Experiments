print("Surya Prasad m\n24BAD119")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

df = pd.read_csv("/home/surya/programs/python-learning/ML lab/bottle.csv", low_memory=False)
df.columns
df1 = pd.read_csv("/home/surya/programs/python-learning/ML lab/cast.csv", low_memory=False)
df1.columns
data = pd.merge(df, df1, on='Cst_Cnt', how='left')
data.head()
target_col = 'T_degC'  

X = data.drop(columns=[target_col]).select_dtypes(include=[np.number])
y = data[target_col]

imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

y = y.fillna(y.mean())
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred = lr_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("--- Baseline Linear Regression Performance ---")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) 
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs Predicted')
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6, color='steelblue', edgecolor='k')
plt.axhline(0, color='red', linestyle='--', lw=2, label='Zero Error')

sns.regplot(x=y_pred, y=residuals, scatter=False, color='darkorange')

plt.xlabel('Predicted Temperature (°C)', fontsize=12)
plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
plt.title('Residuals vs Predicted Values (Homoscedasticity Check)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
print("--- Model Optimization ---")

selector = SelectKBest(score_func=f_regression, k=5) 
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

print(f"Selected Features: {X.columns[selector.get_support()].tolist()}")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_selected, y_train)
y_pred_ridge = ridge_model.predict(X_test_selected)
print(f"Ridge RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.4f}")
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_selected, y_train)
y_pred_lasso = lasso_model.predict(X_test_selected)
print(f"Lasso RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lasso)):.4f}")