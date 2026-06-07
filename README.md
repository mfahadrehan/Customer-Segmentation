# 🛒 Customer Segmentation Using K-Means Clustering

> Unsupervised ML project that segments mall customers into 5 distinct groups based on age, income, and spending behavior — built during AI/ML Internship at DevelopersHub Corporation.

## 📊 Results
- **5 clear customer segments** identified using K-Means
- Elbow Method used to determine optimal number of clusters
- Silhouette score used to validate cluster quality
- Segments: Low income/low spenders, High income/high spenders, Young high spenders, Conservative spenders, Middle-ground customers

## 🛠️ Tech Stack
- **ML:** Scikit-learn (KMeans), LabelEncoder
- **EDA:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Dataset:** Mall Customers CSV (CustomerID, Genre, Age, Annual Income, Spending Score)

## 📁 Project Structure
```
Customer_Segmentation/
├── app.py                                        # Main script
├── Customer_Segmentation_Using_Clustering.ipynb  # Full analysis notebook
└── Mall_Customers.csv                            # Dataset
```

## ⚙️ How to Run
```bash
git clone https://github.com/mfahadrehan/Customer-Segmentation.git
cd Customer-Segmentation
pip install pandas numpy matplotlib seaborn scikit-learn
python app.py
# Or open the Jupyter notebook for full walkthrough
jupyter notebook Customer_Segmentation_Using_Clustering.ipynb
```

## 🔍 Methodology
1. **EDA** — Explored distributions, pairplots, and gender breakdown
2. **Preprocessing** — Label encoded Gender, selected 3 clustering features
3. **Elbow Method** — Tested k=1 to 10, identified optimal k=5
4. **K-Means Clustering** — Fit model and assigned cluster labels
5. **Visualization** — Scatter plots showing clear segment separation

## 🧩 Challenges Solved
- Resolved MKL memory warning on Windows using `OMP_NUM_THREADS=1`
- Correctly scaled features for meaningful cluster separation
