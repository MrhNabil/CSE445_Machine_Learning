# CSE445: Machine Learning – Project on Housing Price Prediction

Welcome to the repository for the course project of **CSE445: Machine Learning**. This repository contains a complete machine learning pipeline implemented in Python to predict and classify housing prices using the **Boston Housing Dataset**. The project explores various ML techniques and evaluation metrics with a focus on model performance and insights.

---

## 🔍 Project Objective

As a Machine Learning Engineer for a real estate firm in the Greater Boston area, the task is to build a predictive model for estimating housing prices using the **Boston Housing Dataset**. This involves:

- Data preprocessing and visualization  
- Training and evaluation of regression models  
- Converting regression into classification  
- Performance evaluation for classifiers  
- Applying ensemble learning methods  
- Analyzing key insights and limitations

---

## 📁 Repository Structure

```

CSE445\_Machine\_Learning/
│
├── Assignment\_1.ipynb        # Complete Jupyter notebook with all tasks
├── datasets/                 # Dataset used (Boston Housing)
├── models/                   # Trained model files (optional)
├── requirements.txt          # List of dependencies
└── README.md                 # Project documentation (this file)

````

---

## ⚙️ Installation Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/MrhNabil/CSE445_Machine_Learning.git
cd CSE445_Machine_Learning
````

2. **(Optional) Create a Virtual Environment**

```bash
python -m venv venv
```

* On Windows:

  ```bash
  venv\Scripts\activate
  ```

* On macOS/Linux:

  ```bash
  source venv/bin/activate
  ```

3. **Install Required Dependencies**

```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**

```bash
jupyter notebook
```

---

## 🧠 Project Breakdown

### 🔹 1. Data Preprocessing & EDA (15 pts)

* Loaded Boston Housing dataset using `sklearn.datasets`.
* Checked and handled missing values.
* Performed exploratory data analysis (EDA) with visualizations.
* Applied feature scaling (`StandardScaler`) and one-hot encoding.
* Included detailed comments explaining all methods used.

### 🔹 2. Regression Models (25 pts)

* Trained the following models:

  * Linear Regression
  * Decision Tree Regressor
  * Random Forest Regressor
  * Support Vector Regressor (SVR)
* Evaluated using:

  * Mean Squared Error (MSE)
  * Mean Absolute Error (MAE)
  * R² Score
* Hyperparameter tuning using `GridSearchCV` and `RandomizedSearchCV`.

### 🔹 3. Regression to Classification (15 pts)

* Transformed the continuous target variable into three price categories:

  * Low
  * Medium
  * High
* Trained classifiers:

  * Logistic Regression
  * Random Forest Classifier
  * SVM Classifier

### 🔹 4. Classification Metrics (15 pts)

* Calculated:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
* Plotted ROC curves and computed AUC scores for each model.

### 🔹 5. Ensemble Learning (15 pts)

* Applied:

  * Bagging (e.g., `BaggingClassifier`)
  * Boosting (Gradient Boosting, AdaBoost, LightGBM)
  * Stacking Classifier
* Evaluated accuracy and feature importance for ensemble models.

### 🔹 6. Insights & Conclusion (15 pts)

* Compared performance between regression and classification tasks.
* Summarized:

  * Best models and metrics
  * Observations on feature importance
  * Limitations and potential improvements

---

## ✅ Requirements

* Python 3.8+
* Jupyter Notebook / Jupyter Lab
* Required libraries listed in `requirements.txt`:

  * numpy
  * pandas
  * matplotlib
  * seaborn
  * scikit-learn
  * lightgbm

---

## 📌 References

* [Boston Housing Dataset - scikit-learn](https://scikit-learn.org/1.0/modules/generated/sklearn.datasets.load_boston.html)
* [California Housing Example (Colab)](https://colab.research.google.com/drive/1O1r4V8CtDv9phrsYfp8ct9sWR74aArvr?usp=sharing)

---

## 📬 Contact

**Nabil**
GitHub: [@MrhNabil](https://github.com/MrhNabil)
Project Repository: [CSE445\_Machine\_Learning](https://github.com/MrhNabil/CSE445_Machine_Learning)


