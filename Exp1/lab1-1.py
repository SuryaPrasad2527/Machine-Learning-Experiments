print("Surya Prasad m\n24BAD119")
import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv('/home/surya/programs/python-learning/ML lab/data.csv')
df.head()
df.tail()
df.info()
df.describe()
df.isnull().sum()

# Remove cancelled orders and nulls
df = df [df['Quantity'] > 0]
df['TotalSales'] = df['Quantity']*df['UnitPrice']
top_products = df.groupby('Description') ['TotalSales'].sum().head(10)

#Bar chart
top_products.plot(kind='bar')
plt.title("Top 10 Products by Sales")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.show()

#Line chart
top_products.plot(kind='line')
plt.title("Sales Trend of Top Products")
plt.show()
