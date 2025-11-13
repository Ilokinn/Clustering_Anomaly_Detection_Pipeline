# Clustering & Anomaly Detection Pipeline

A comprehensive data science project exploring clustering algorithms and anomaly detection 
techniques on high-dimensional datasets.

## 🎯 Project Overview

This project investigates:
- **Clustering** performance across different algorithms (K-Means, DBSCAN, Spectral Clustering)
- Impact of dimensionality reduction (PCA, UMAP) on clustering quality
- **Anomaly Detection** using ensemble methods and deep learning autoencoders
- Comparative analysis with internal/external validation metrics

## 📊 Datasets

- **Hi-Seq**: Gene expression data (20,531 features, 801 samples)
  - Multiple cancer types classification
  - High-dimensional exploration case study
  
- **ECG**: Cardiac signal recordings (141 features, 4,998 samples)
  - Normal vs. abnormal pattern detection
  - Time-series anomaly detection benchmark

## 🏗️ Project Structure
├── src/
│   ├── models.py              # Algorithm implementations
│   ├── preprocess.py          # Data cleaning & transformation
│   ├── train.py               # Training pipelines
│   └── utils.py               # Metrics & utilities
├── notebooks/
│   └── analysis.ipynb         # Main analysis & visualization
├── results/
│   ├── clustering_results.csv
│   └── anomaly_results.csv
├── requirements.txt
└── README.md
## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
jupyter notebook notebooks/analysis.ipynb
```

## 📈 Methodology

### Experimental Protocol
- **10 independent runs** per algorithm
- Different random initialization each run
- Results reported as (μ ± σ)

### Clustering Evaluation
**Internal Metrics** (ground-truth independent):
- Silhouette Coefficient
- Davies-Bouldin Index
- Calinski-Harabasz Index

**External Metrics** (with labels):
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)

### Anomaly Detection Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC score
- Training time & memory usage

## 🔬 Methods Implemented

### Clustering
1. **K-Means** - Partitioning approach
2. **DBSCAN** - Density-based clustering
3. **Spectral Clustering** - Graph-based method

Each tested with:
- Full feature set
- Reduced dimensions (PCA/UMAP to 100 features)

### Anomaly Detection
1. **Isolation Forest** - Ensemble isolation method
2. **Standard Autoencoder** - Reconstruction-based baseline
3. **Advanced Autoencoder Variants**:
   - Denoising Autoencoder
   - Variational Autoencoder (VAE)
   - Sparse Autoencoder
   - Monte Carlo Dropout Autoencoder

## 📊 Key Results

[Results summary to be added after analysis]

## 🛠️ Technologies

- Python 3.8+
- scikit-learn
- TensorFlow/PyTorch
- Pandas, NumPy
- Matplotlib, Seaborn
- UMAP for dimensionality reduction

## 📝 Notes

- Data preprocessing ensures normalization for fair comparisons
- Hyperparameter tuning performed on validation sets
- Reproducibility maintained through random seed management

## 📜 License

MIT License - feel free to use for learning and research.
