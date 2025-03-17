# K-Nearest Neighbors (K-NN) Classification for Wine Quality Prediction

**Author:** Peng Gao  
**Course:** Senior Design I  

---

## **1. Introduction**
The objective of this assignment is to implement a **K-Nearest Neighbors (K-NN) classifier** to predict **wine quality** based on chemical properties. The dataset `winequality-white.csv` is used for training and evaluation, with a **binary classification**:
- **Good wine (`quality > 5`) → 1**
- **Bad wine (`quality <= 5`) → 0**

This report covers **data preprocessing, model implementation, hyperparameter tuning, and evaluation**.

---

## **2. Data Preprocessing**

### **2.1 Dataset Overview**
The dataset contains **4898 samples** with **11 numerical features**:
- `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`

The `quality` column is transformed into a **binary target variable**.

### **2.2 Data Integrity Checks**
- **Missing Values:** None found.
- **Duplicate Entries:** Removed if found.
- **Feature Summary:** Mean, standard deviation, and quartiles computed.

### **2.3 Feature Selection**
- **Pair plots generated** to identify correlated features.
- **Highly correlated features dropped** to avoid redundancy.

### **2.4 Data Splitting & Scaling**
- Data shuffled before splitting.
- **80% training, 20% testing split** using a custom `partition()` function.
- **Feature standardization applied**: `(x - mean) / std_dev`

---

## **3. K-NN Model Implementation**
A **K-Nearest Neighbors (K-NN) classifier** was implemented using **Scikit-Learn**. The classifier includes:
- **Distance Functions:** Euclidean & Manhattan
- **Weighting Methods:** Uniform & Distance-based
- **Hyperparameter Selection:** Different `k` values tested

### **3.1 `fit()` Method**
Stores training data and model parameters.

### **3.2 `predict()` Method**
- Computes distances to all training samples.
- Selects the **k nearest neighbors**.
- Applies **weighted voting** for classification.

---

## **4. Model Evaluation**
The classifier is evaluated using multiple metrics:

### **4.1 Hyperparameter Testing**
Tested combinations:
- `k = [1, 5, 9, 11]`
- `Distance = [Euclidean, Manhattan]`
- `Weights = [uniform, distance]`

### **4.2 Performance Metrics**
| k  | Metric    | Weight  | Accuracy | F1 Score |
|----|----------|---------|----------|----------|
| 5  | Manhattan | Distance | **0.8357** | **0.8792** |
| 11 | Manhattan | Distance | 0.8337 | 0.8780 |
| 9  | Euclidean | Distance | 0.8316 | 0.8775 |
| 5  | Euclidean | Distance | 0.8235 | 0.8708 |
| 5  | Euclidean | Uniform  | 0.8048 | 0.8545 |
| 1  | Manhattan | Uniform  | 0.8016 | 0.8589 |
| 9  | Euclidean | Uniform  | 0.7699 | 0.8255 |
| 11 | Euclidean | Uniform  | 0.7602 | 0.8235 |

**Best Parameters:**  
- **k = 5**  
- **Metric = Manhattan**  
- **Weighting = Distance**

### **4.3 Final Model Performance**
- **Final Accuracy:** **0.8357**
- **Final Precision:** **0.8694**
- **Final Recall:** **0.8892**
- **Final F1 Score:** **0.8792**

### **4.4 Confusion Matrix**
The **Confusion Matrix** for the best-performing model (`k=5`, Manhattan distance, distance-weighted) is:  
<pre>
            Predicted: 0    Predicted: 1  
Actual: 0       233             88  
Actual: 1       73              586  
</pre>

### **4.5 ROC Curve & AUC Score**
- The **ROC Curve** was plotted.
- **AUC Score: [Insert AUC Value]**

### **4.6 Precision-Recall Curve**
- Precision-Recall curve was plotted.
- Helped evaluate performance on imbalanced classes.

---

## **5. Conclusion**
- The **best-performing model** used `k = 5`, `metric = Manhattan`, and `weighting = Distance`.  
- Standardization **improved accuracy** significantly.  
- **Manhattan distance performed better** than Euclidean in this dataset.  
- Future improvements could include **cross-validation and additional feature selection**.

