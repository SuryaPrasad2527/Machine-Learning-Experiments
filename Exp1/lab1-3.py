print("Surya Prasad m\n24BAD119")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_csv('/home/surya/programs/python-learning/ML lab/Housing.csv')
df.info()
df.head()
df.isnull().sum()

plt.scatter(df['area'], df['price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()
