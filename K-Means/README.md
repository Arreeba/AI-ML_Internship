# Customer Segmentation using K-Means Clustering

## Overview
This project applies **K-Means Clustering** to the Mall Customers dataset to identify groups of customers with similar purchasing behavior. The clustering is performed using **Annual Income** and **Spending Score**, enabling meaningful customer segmentation for targeted marketing and business decision-making.

## Dataset
- **Source:** Mall Customers Dataset
- **Samples:** 200 customers
- **Features Used:**
  - Annual Income (k$)
  - Spending Score (1–100)
- No missing values

## Approach
1. Loaded and explored the dataset.
2. Selected **Annual Income** and **Spending Score** as clustering features.
3. Standardized the features using **StandardScaler** to ensure equal contribution to distance calculations.
4. Applied the **Elbow Method** (WCSS/Inertia) to determine the optimal number of clusters.
5. Trained the final **K-Means** model using **K = 5**.
6. Assigned cluster labels to each customer.
7. Visualized the clusters along with their centroids.
8. Evaluated clustering quality using the **Silhouette Score**.
9. Analyzed the average characteristics of each cluster and assigned meaningful business labels.

## Results

| Metric | Value |
|--------|-------|
| Optimal Number of Clusters | 5 |
| Silhouette Score | 0.555 |

The Elbow Method suggested **5 clusters**, while the Silhouette Score of **0.555** indicates reasonably well-separated customer groups with moderate cluster quality.

## Customer Segments

| Cluster | Segment | Characteristics |
|---------|---------|-----------------|
| 0 | Standard | Medium income, medium spending |
| 1 | Target | High income, high spending |
| 2 | Careless | Low income, high spending |
| 3 | Careful | High income, low spending |
| 4 | Sensible | Low income, low spending |

These segments can support marketing strategies such as:
- Rewarding high-value customers
- Retaining standard customers
- Promoting products to careful spenders
- Understanding low-income spending patterns

## Key Learnings
- K-Means is an unsupervised learning algorithm that groups similar data points without using labels.
- Feature scaling is essential because K-Means relies on Euclidean distance.
- The Elbow Method helps estimate the optimal number of clusters by analyzing WCSS.
- The Silhouette Score measures how well-separated and cohesive the clusters are.
- Cluster centroids represent the average characteristics of each customer group.
- Business interpretation is an important final step, turning numerical clusters into actionable customer segments.

## How to Run
1. Open `kmeans_customer_segmentation.ipynb` in Google Colab or Jupyter Notebook.
2. Run all cells in order.
3. The dataset is automatically loaded from the provided GitHub URL.
4. The notebook generates:
   - Elbow Method plot
   - Customer cluster visualization
   - Silhouette Score
   - Cluster profiles and business segment labels
