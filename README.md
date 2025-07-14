```markdown
# CSE445: Machine Learning Course Projects - Boston Housing Price Prediction

Welcome to the repository for CSE445: Machine Learning! This repository houses the project completed as part of the course, focusing on building a predictive model for housing prices in the Greater Boston area using the **Boston Housing Dataset**. As an ML Engineer for a real estate firm, the goal was to apply comprehensive machine learning techniques, from data preprocessing to advanced ensemble learning, to determine the best-performing model.

---

## Project Overview

This project features a detailed implementation in Python using Jupyter notebooks, covering a wide array of core ML topics:

* **Supervised Learning:** Deep dives into both **regression** (predicting housing prices) and **classification** (categorizing housing prices) problems.
* **Data Preprocessing & Visualization:** Essential steps for preparing and understanding the dataset, including handling missing values, feature scaling, and one-hot encoding.
* **Algorithm Implementation:** Utilization of popular machine learning libraries like `scikit-learn` for various models.
* **Model Evaluation:** Comprehensive comparison of model performance using appropriate metrics for both regression (MSE, MAE, R²) and classification (Accuracy, Precision, Recall, F1-score, ROC-AUC).
* **Hyperparameter Tuning:** Optimization of model performance using `GridSearchCV` and `RandomizedSearchCV`.
* **Ensemble Learning:** Application of advanced techniques like Bagging, Boosting (Gradient Boosting, AdaBoost, LightGBM), and Stacking to enhance model accuracy.
* **Feature Importance:** Identification of the most impactful features in predicting housing prices.

---

## Repository Structure

The repository is organized for easy navigation:

```

CSE445\_Machine\_Learning/
│
├── Project\_Boston\_Housing.ipynb   \# Main project notebook for Boston Housing Prediction
├── datasets/                      \# Data files used in the project (Boston Housing Dataset)
├── models/                        \# Saved machine learning models (if any)
├── README.md                      \# This file
└── requirements.txt               \# Python dependencies

````

---

## Installation

To get this project up and running on your local machine, follow these simple steps:

### 1. Clone the Repository:

```bash
git clone [https://github.com/MrhNabil/CSE445_Machine_Learning.git](https://github.com/MrhNabil/CSE445_Machine_Learning.git)
cd CSE445_Machine_Learning
````

### 2\. (Optional) Create and Activate a Virtual Environment:

It's highly recommended to use a virtual environment to manage dependencies and avoid conflicts with other Python projects.

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3\. Install Required Packages:

All necessary Python libraries are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4\. Launch Jupyter:

Once everything is installed, you can open the notebook using Jupyter Notebook or Jupyter Lab:

```bash
jupyter notebook
```

This command will open a new tab in your web browser, displaying the Jupyter interface.

-----

## Usage

1.  Open the `Project_Boston_Housing.ipynb` notebook within the Jupyter interface.
2.  Run the cells sequentially to execute the code and see the results.
3.  The notebook contains detailed instructions, comments, and explanations for each task, guiding you through the specific concepts and implementations.

-----

## Project Tasks & Implementation Details

### 1\. Data Preprocessing and Early Data Analysis (EDA)

  * **Loading the Dataset:** The Boston Housing Dataset was loaded using `sklearn.datasets.load_boston`.
  * **Handling Missing Values:** Although the `load_boston` dataset is typically clean, standard checks and potential imputation strategies (if needed for other datasets) would be discussed.
  * **Visualizing Key Features:** Exploratory Data Analysis (EDA) was performed to understand the distribution of features and their relationship with the target variable (housing price) through various plots (histograms, scatter plots).
  * **Feature Scaling:** Applied techniques like **Standardization** (`StandardScaler`) to normalize the range of independent variables. This is crucial for models sensitive to feature scales (e.g., SVM, Linear Regression).
  * **One-Hot Encoding:** Categorical features (if any in the extended Boston Housing Dataset or similar datasets) would be transformed into a numerical format using **One-Hot Encoding** (`OneHotEncoder`) to prevent models from assuming ordinal relationships.
  * **Code Explanation:** Every method and function used throughout this stage is thoroughly commented and explained within the notebook, detailing its purpose and impact on the data.

### 2\. Train and Evaluate Regression Models

  * **Model Training:**
      * **Linear Regression:** A baseline linear model for predicting continuous housing prices.
      * **Decision Tree Regressor:** A non-linear model capable of capturing complex relationships.
      * **Random Forest Regressor:** An ensemble method that combines multiple decision trees to improve accuracy and reduce overfitting.
      * **Support Vector Machine (SVR):** A powerful model for both linear and non-linear regression tasks, especially effective in high-dimensional spaces.
  * **Performance Comparison:** Models were evaluated using:
      * **Mean Squared Error (MSE):** Measures the average squared difference between estimated and actual values.
      * **Mean Absolute Error (MAE):** Measures the average absolute difference between estimated and actual values, less sensitive to outliers than MSE.
      * **R² Score:** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables, indicating the goodness of fit.
  * **Hyperparameter Tuning:**
      * **GridSearchCV:** Exhaustively searches over a specified parameter grid for the best combination.
      * **RandomizedSearchCV:** Samples a fixed number of parameter settings from specified distributions, often more efficient for large search spaces. Each parameter tuning step is accompanied by detailed explanations of the chosen parameters and the rationale behind their selection.

### 3\. Convert Regression to Classification

  * **Price Categorization:** The continuous housing price variable was transformed into discrete categories ("Low", "Medium", "High") based on **percentiles**. This process involves defining thresholds to segment the price range.
  * **Classifier Training:**
      * **Logistic Regression:** A linear model adapted for binary or multi-class classification.
      * **Random Forest Classifier:** An ensemble method known for its robustness and accuracy in classification tasks.
      * **Support Vector Machine (SVC):** A powerful classifier, particularly effective in high-dimensional spaces with clear margins of separation.

### 4\. Calculate Classification Metrics

  * **Performance Metrics:** For each classifier, the following metrics were computed:
      * **Accuracy:** The proportion of correctly classified instances.
      * **Precision:** The proportion of positive identifications that were actually correct.
      * **Recall (Sensitivity):** The proportion of actual positives that were correctly identified.
      * **F1-score:** The harmonic mean of precision and recall, providing a balanced measure.
  * **ROC Curve and AUC Score:**
      * **Receiver Operating Characteristic (ROC) Curve:** A graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.
      * **Area Under the Curve (AUC):** Measures the entire two-dimensional area underneath the entire ROC curve, providing an aggregate measure of performance across all possible classification thresholds.

### 5\. Apply Ensemble Learning

  * **Ensemble Techniques:**
      * **Bagging (e.g., BaggingClassifier with Decision Trees):** Reduces variance and helps to avoid overfitting by training multiple models independently on different subsets of the training data and averaging their predictions.
      * **Boosting:** Sequentially builds models, where each new model corrects errors made by the previous ones.
          * **Gradient Boosting (e.g., `GradientBoostingClassifier`):** Builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions.
          * **AdaBoost (Adaptive Boosting):** Focuses on instances that were misclassified by earlier models.
          * **LightGBM (`lightgbm.LGBMClassifier`):** A gradient boosting framework that uses tree-based learning algorithms, known for its high performance and speed.
      * **Stacking:** Combines multiple models using another model (a meta-learner) to make predictions based on the predictions of the individual models.
  * **Feature Importance:** For tree-based models (e.g., Random Forest, Gradient Boosting), **feature importance scores** were extracted and visualized to identify which features had the most significant impact on the model's predictions. This provides valuable insights into the underlying drivers of housing prices.

### 6\. Insights & Conclusion

  * **Model Performance Analysis:** A detailed analysis comparing the performance of all regression and classification models was conducted. This included discussing their strengths, weaknesses, and suitability for the Boston Housing Dataset.
  * **Key Takeaways:** A summary of critical insights gained from each of the five steps:
      * **Data Preprocessing:** Importance of cleaning, scaling, and encoding data for model performance.
      * **Regression Modeling:** Comparison of different regression algorithms and their respective error metrics, highlighting which models performed best for continuous price prediction.
      * **Classification Conversion:** The methodology and implications of transforming a regression problem into a classification one, and the new insights gained.
      * **Classification Metrics:** The importance of choosing appropriate metrics (beyond just accuracy) for a balanced evaluation of classification models, especially in imbalanced datasets.
      * **Ensemble Learning:** The effectiveness of ensemble methods in boosting model accuracy and robustness, along with the value of feature importance for interpretability.
  * **Limitations:** Discussion of any limitations encountered, such as dataset size, potential biases, or computational constraints, and suggestions for future improvements.

-----

## Requirements

To ensure compatibility and smooth execution, you'll need:

  * Python 3.8+
  * Jupyter Notebook (or Jupyter Lab)
  * The Python libraries listed in `requirements.txt`, including `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, and `lightgbm`.

-----

## Contact

For any questions, feedback, or collaborations, feel free to reach out:

**Nabil**
GitHub: [@MrhNabil](https://www.google.com/search?q=https://github.com/MrhNabil)
Repository Link: [https://github.com/MrhNabil/CSE445\_Machine\_Learning](https://github.com/MrhNabil/CSE445_Machine_Learning)

```
```
