# scikit-learn Cookbook — Third Edition

> **Reproduction of *scikit-learn Cookbook, Third Edition* by John Sukup (Packt, 2025)**
> Over 80 recipes for machine learning in Python with scikit-learn

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)

---

## 📖 About This Repository

This repository contains chapter-by-chapter reproductions of the code from *scikit-learn Cookbook, Third Edition*, with added **summaries** and **theoretical explanations** for each topic. Each chapter is presented as a standalone Jupyter notebook that you can run, modify, and experiment with.

The goal is to provide a complete, educational reference for scikit-learn v1.5 — from basic API conventions through to production deployment.

---

## 🗂️ Repository Structure

```
scikit-learn-Cookbook-Third-Edition/
│
├── README.md
│
├── Chapter_01_Common_Conventions_and_API_Elements_of_scikit-learn/
│   └── notebook.ipynb
│
├── Chapter_02_Pre-Model_Workflow_and_Data_Preprocessing/
│   └── notebook.ipynb
│
├── Chapter_03_Dimensionality_Reduction_Techniques/
│   └── notebook.ipynb
│
├── Chapter_04_Building_Models_with_Distance_Metrics_and_Nearest_Neighbors/
│   └── notebook.ipynb
│
├── Chapter_05_Linear_Models_and_Regularization/
│   └── notebook.ipynb
│
├── Chapter_06_Advanced_Logistic_Regression_and_Extensions/
│   └── notebook.ipynb
│
├── Chapter_07_Support_Vector_Machines_and_Kernel_Methods/
│   └── notebook.ipynb
│
├── Chapter_08_Tree-Based_Algorithms_and_Ensemble_Methods/
│   └── notebook.ipynb
│
├── Chapter_09_Text_Processing_and_Multiclass_Classification/
│   └── notebook.ipynb
│
├── Chapter_10_Clustering_Techniques/
│   └── notebook.ipynb
│
├── Chapter_11_Novelty_and_Outlier_Detection/
│   └── notebook.ipynb
│
├── Chapter_12_Cross-Validation_and_Model_Evaluation_Techniques/
│   └── notebook.ipynb
│
└── Chapter_13_Deploying_scikit-learn_Models_in_Production/
    └── notebook.ipynb
```

---

## 📚 Chapter Summaries

### Chapter 1 — Common Conventions and API Elements of scikit-learn
Introduces the core design philosophy of scikit-learn: **estimators**, **transformers**, and **pipelines**. All scikit-learn objects follow a consistent interface using `fit()`, `predict()`, and `transform()`. This chapter explains how to use `Pipeline` for workflow automation, `GridSearchCV` for hyperparameter tuning, and how to build custom estimators by subclassing `BaseEstimator`.

### Chapter 2 — Pre-Model Workflow and Data Preprocessing
Covers the most critical step in ML: preparing your data. Topics include handling missing values with `SimpleImputer`, `KNNImputer`, and `IterativeImputer`; feature scaling with `StandardScaler`, `MinMaxScaler`, and `RobustScaler`; encoding categorical variables with `OneHotEncoder` and `OrdinalEncoder`; and building full preprocessing pipelines with `ColumnTransformer`. Key principle: *garbage in, garbage out*.

### Chapter 3 — Dimensionality Reduction Techniques
Addresses the *curse of dimensionality* with three techniques: **PCA** (unsupervised, preserves variance), **LDA** (supervised, maximizes class separability), and **t-SNE** (non-linear, visualization only). Includes practical guidance on scree plots, choosing the number of components, and the impact of dimensionality reduction on downstream model performance.

### Chapter 4 — Building Models with Distance Metrics and Nearest Neighbors
Explores K-Nearest Neighbors (KNN), a lazy learner that classifies based on the K most similar training points. Covers distance metrics (Euclidean, Manhattan, Minkowski, Chebyshev), the effect of k on bias-variance tradeoff, weighted KNN, and hyperparameter tuning. Emphasizes that **feature scaling is mandatory** for distance-based models.

### Chapter 5 — Linear Models and Regularization
Covers linear and regularized regression. **Ridge** (L2) shrinks coefficients toward zero. **Lasso** (L1) performs automatic feature selection by driving some coefficients to exactly zero. **ElasticNet** combines both penalties. The `RidgeCV`, `LassoCV`, and `ElasticNetCV` variants automatically select the best regularization strength via cross-validation.

### Chapter 6 — Advanced Logistic Regression and Extensions
Despite the name, logistic regression is a **classification** model that outputs probabilities via the sigmoid function. This chapter covers binary and multiclass strategies (OvR, OvO, multinomial), regularization with `C` parameter, multilabel classification, and a comprehensive set of evaluation metrics including ROC-AUC, precision-recall curves, and F1 score.

### Chapter 7 — Support Vector Machines and Kernel Methods
SVMs find the **maximum margin hyperplane** that best separates classes. The **kernel trick** (RBF, polynomial) extends SVMs to non-linear boundaries without explicitly computing high-dimensional features. Covers the `C` and `gamma` hyperparameters, SVM for regression (SVR), and decision boundary visualization. Feature scaling is critical for SVM performance.

### Chapter 8 — Tree-Based Algorithms and Ensemble Methods
Decision trees recursively split the feature space using Gini impurity or entropy. **Random Forests** reduce overfitting via bagging and random feature selection. **Gradient Boosting** sequentially corrects errors of previous trees for high accuracy. Covers feature importance, hyperparameter tuning, and a comprehensive comparison of ensemble methods (AdaBoost, Extra Trees, GBM, HistGBM).

### Chapter 9 — Text Processing and Multiclass Classification
Text data must be converted to numbers before ML can be applied. Covers `CountVectorizer` (bag of words), `TfidfVectorizer` (importance-weighted counts), and n-grams for capturing word sequences. Demonstrates a full text classification pipeline using the 20 Newsgroups dataset with Multinomial Naive Bayes and SGD classifier. Evaluates models with confusion matrices and classification reports.

### Chapter 10 — Clustering Techniques
Unsupervised learning for finding natural groupings. **K-Means** minimizes within-cluster sum of squares — use the elbow method or silhouette score to choose K. **Hierarchical clustering** builds a dendrogram of nested clusters. **DBSCAN** handles arbitrary shapes and automatically identifies noise points. Covers cluster evaluation metrics (silhouette score, Davies-Bouldin index, ARI) for both labeled and unlabeled data.

### Chapter 11 — Novelty and Outlier Detection
Identifies anomalous data points that deviate from the norm — critical for fraud detection, quality control, and intrusion detection. **Isolation Forest** isolates anomalies with random splits. **LOF** compares local density to neighbors. **One-Class SVM** learns a boundary around normal data. **Elliptic Envelope** assumes Gaussian distribution. Discusses the `contamination` parameter and strategies for handling detected outliers.

### Chapter 12 — Cross-Validation and Model Evaluation Techniques
A single train/test split gives unreliable estimates. **K-Fold CV** averages performance across multiple splits. **Stratified K-Fold** preserves class distributions. **TimeSeriesSplit** prevents future leakage in temporal data. **Learning curves** diagnose bias/variance. **Validation curves** reveal optimal hyperparameter ranges. **Nested CV** gives unbiased estimates when also tuning hyperparameters.

### Chapter 13 — Deploying scikit-learn Models in Production
Training is only the beginning. Covers model serialization with `joblib`, versioning with metadata, batch prediction, monitoring for data drift, and deployment pipeline automation with validation thresholds. Provides a FastAPI skeleton for serving real-time predictions. Discusses MLOps concepts: continuous training, model registries, and integration with tools like MLflow.

---

## ⚙️ Setup and Installation

### Prerequisites
- Python >= 3.9
- pip or conda

### Installation

```bash
# Clone this repository
git clone https://github.com/ranggatimotius/scikit-learn-Cookbook-Third-Edition.git
cd scikit-learn-Cookbook-Third-Edition

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install scikit-learn>=1.5 numpy pandas matplotlib seaborn jupyter joblib scipy
```

### Running Notebooks

```bash
jupyter notebook
```

Then navigate to the chapter folder and open `notebook.ipynb`.

---

## 🛠️ Key Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| scikit-learn | ≥ 1.5 | Core ML library |
| numpy | ≥ 1.24 | Numerical computing |
| pandas | ≥ 2.0 | Data manipulation |
| matplotlib | ≥ 3.7 | Visualization |
| seaborn | ≥ 0.12 | Statistical visualization |
| scipy | ≥ 1.10 | Scientific computing |
| joblib | ≥ 1.3 | Model serialization |

---

## 📌 Learning Path

If you're new to scikit-learn, follow this recommended order:

1. **Start here**: Chapters 1–2 (API + Preprocessing)
2. **Core supervised learning**: Chapters 4–8 (KNN, Linear, Logistic, SVM, Trees)
3. **Evaluation**: Chapter 12 (Cross-validation)
4. **Advanced topics**: Chapters 3, 9, 10, 11 (Reduction, Text, Clustering, Outliers)
5. **Production**: Chapter 13 (Deployment)

---

## 📝 Notes

- All notebooks are self-contained and can be run independently.
- Code is written for **scikit-learn v1.5** — some APIs may differ in older versions.
- Theoretical explanations use LLM assistance for clarity, as permitted by the assignment guidelines.
- Each notebook follows the structure: Summary → Theory → Code → Key Takeaways.

---

## 🔗 References

- [Official scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Book GitHub Repository (Packt)](https://github.com/PacktPublishing/scikit-learn-Cookbook-Third-Edition)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

*Reproduced for educational purposes as part of a machine learning course assignment.*
