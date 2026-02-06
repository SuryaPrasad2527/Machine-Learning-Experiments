print("Surya Prasad m\n24BAD119")
import pandas as pd
import matplotlib.pyplot as plt
df= pd.read_csv('/home/surya/programs/python-learning/ML lab/diabetes.csv')
df.info()
df.describe()
df.isnull().sum()
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns] = df [columns].replace(0, pd.NA)
df.isnull().sum()

plt.hist(df['Glucose'].dropna())
plt.title("Glucose Level Distribution")
plt.xlabel("Glucose")
plt.show()

plt.boxplot(df['Age'])
plt.title("Age Distribution")
plt.show()

df.groupby ('Outcome').mean()
