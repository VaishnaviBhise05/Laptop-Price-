# Laptop-Price-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('laptop_prices.csv')
df
print(df.info())
print(df.head())
print(df.describe(include='all'))
print("First 5 rows:")
print(df.head())
print(f"\nDataset contains {df.shape[0]} rows and {df.shape[1]} columns")
print("\nInfo:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe(include='all'))
print("\nMissing Values:")
print(df.isnull().sum())

plt.figure(figsize=(10, 6))
sns.histplot(df['Price_euros'], bins=30, kde=True)
plt.title('Distribution of Laptop Prices (Euros)')
plt.xlabel('Price (Euros)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 8))
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

company_counts = df['Company'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(company_counts, labels=company_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Laptops by Company')
plt.axis('equal')  
plt.tight_layout()
plt.show()

os_counts = df['OS'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(os_counts, labels=os_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Operating Systems')
plt.axis('equal')
plt.tight_layout()
plt.show()


touchscreen_counts = df['Touchscreen'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(touchscreen_counts, labels=['No Touchscreen', 'Touchscreen'], autopct='%1.1f%%', 
        colors=['lightblue', 'lightgreen'], startangle=90)
plt.title('Touchscreen Feature Distribution')
plt.axis('equal')
plt.tight_layout()
plt.show()

print("Missing values per column:")
print(df.isnull().sum())

categorical_columns = df.select_dtypes(include='object').columns
for col in categorical_columns:
    print(f"\nUnique values in {col}:")
    print(df[col].unique())

avg_price_os = df.groupby('OS')['Price_euros'].mean()
print("\nAverage price by OS:")
print(avg_price_os)

storage_type_counts = df['PrimaryStorageType'].value_counts()
print("\nPrimary Storage Type Counts:")
print(storage_type_counts)

top_expensive = df.sort_values(by='Price_euros', ascending=False).head(5)
print("\nTop 5 Most Expensive Laptops:")
print(top_expensive[['Company', 'Product', 'Price_euros']])

filtered_laptops = df[(df['Ram'] > 16) & (df['PrimaryStorageType'] == 'SSD')]
print(f"\nLaptops with >16GB RAM and SSD: {filtered_laptops.shape[0]} found")

df['TotalStorage'] = df['PrimaryStorage'] + df['SecondaryStorage']

df['HighResolution'] = (df['ScreenW'] >= 1920) & (df['ScreenH'] >= 1080)

print("\nNew columns 'TotalStorage' and 'HighResolution' added.")
