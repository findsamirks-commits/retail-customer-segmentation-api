# 🛒 Retail Customer Segmentation & RFM Clustering API

Welcome to the **Retail Customer Segmentation API** repository! This project tackles a core operational challenge in retail: moving away from one-size-fits-all marketing by using unsupervised machine learning to automatically group shoppers into behavioral segments.

---

## 🚀 Project Architecture & Workflow

```mermaid
graph TD
    A[Raw Retail Transaction Logs] --> B[RFM Feature Engineering: Recency, Frequency, Monetary]
    B --> C[Feature Scaling: StandardScaler]
    C --> D[Unsupervised Learning: K-Means Clustering]
    D --> E[Serialize Model & Scaler to .joblib]
    E --> F[FastAPI Backend: /segment Endpoint]
    F --> G[Instant Actionable Insights & Employee Workflows]
