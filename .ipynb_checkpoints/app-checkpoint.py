# Customer Segmentation - Mall Dataset

## 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

## 2. Load Dataset
df = pd.read_csv('Mall_Customers.csv')
print(df.head())

## 3. EDA
print(df.info())
print(df.describe())
sns.pairplot(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
plt.show()

# Gender count
sns.countplot(data=df, x='Gender')
plt.show()

## 4. Preprocessing
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

## 5. Elbow Method
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

## 6. Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

## 7. Visualize Clusters
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='Set1')
plt.title('Customer Segments')
plt.show()

## 8. Interpretation
# Add your observations and interpretations here
