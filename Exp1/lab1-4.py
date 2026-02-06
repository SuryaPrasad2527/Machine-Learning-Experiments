print("Surya Prasad m\n24BA0119")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/home/surya/programs/python-learning/ML lab/marketing_campaign.csv'
,sep=None, engine='python')
df.info()
df.head()
df.info()
df.describe()
df.isnull().sum()
df ['Income'].fillna (df['Income'].median(), inplace=True)
df['Age'] = 2024 - df ['Year_Birth']
plt.hist(df['Age'], bins-10)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()
plt.boxplot(df ['Income'])
plt.title("Income Distribution of Customers")
plt.ylabel("Income")
plt.show()
spending_cols= [
'MntWines', 'MntFruits', 'MntMeatProducts',
'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df [spending_cols].sum().plot(kind='bar')
plt.title("Customer Spending Patterns")
plt.ylabel("Total Amount Spent")
plt.show()
