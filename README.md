# Gene Mutation Classification 🧬

## Overview
This notebook builds a **gene mutation classification model** to predict whether a gene is **highly mutated** based on various genomic features. The model uses a **deep learning approach** with TensorFlow/Keras.

## Dataset
The dataset contains frequently mutated genes and includes features such as:
- **Number of mutations**
- **Copy Number Variation (CNV) gains/losses**
- **Affected cases percentages**

The target variable (`high_mutation`) is created based on whether a gene's mutation frequency exceeds a set threshold.

## Steps in the Notebook
### 1️⃣ Data Preprocessing
- Load the dataset using pandas.
- Drop unnecessary columns (`gene_id`, `type`, `annotations`).
- Scale numerical features using `StandardScaler`.
- Create a binary classification target (`high_mutation`).
- Split data into **80% training** and **20% testing**.

### 2️⃣ Model Architecture
A **Neural Network** is built using TensorFlow/Keras:
- **Input Layer:** Matches the number of selected features.
- **Hidden Layers:**
  - First hidden layer: 32 neurons, ReLU activation.
  - Second hidden layer: 16 neurons, ReLU activation.
- **Output Layer:**
  - 1 neuron with sigmoid activation (binary classification).
  - Loss function: **Binary Cross-Entropy**.
  - Optimizer: **Adam**.

### 3️⃣ Training Strategy
- Class imbalance is handled using `compute_class_weight`.
- Trained for **50 epochs** with a batch size of 16.
- Uses **validation data** to monitor performance.

### 4️⃣ Model Evaluation
- Computes **test accuracy**.
- Precision could be improved by tuning hyperparameters or adding more data.

## Possible Improvements
✅ Add more **explanations** for beginners.
✅ Perform **Exploratory Data Analysis (EDA)** (e.g., visualizing mutations).  
✅ Include **Precision, Recall, F1-Score, and Confusion Matrix** for better evaluation.  
✅ Experiment with **hyperparameter tuning** (learning rate, batch size, number of layers).  
✅ Try other models (e.g., **Random Forest, XGBoost**).  

## How to Run the Notebook
1. Install dependencies:  
   ```bash
   pip install numpy pandas tensorflow scikit-learn matplotlib
   ```
2. Load the dataset (`frequently-mutated-genes.csv`).
3. Run all cells in the notebook.

## Conclusion
This project successfully classifies highly mutated genes using a deep learning approach. While the model performs well, improvements in feature engineering, evaluation metrics, and tuning could further enhance performance. 🚀

