# Predicting Breast Cancer Using Logistic Regression from Scratch

## Overview

This project implements **Logistic Regression from scratch** to predict whether a breast cancer tumor is malignant or benign. The dataset includes patient data such as tumor size, lymph node involvement, and menopause status, which are used as features to train the model.

The logistic regression model is built from scratch using gradient descent for optimization. This implementation does not rely on machine learning libraries, offering a deep understanding of how logistic regression works.

## Project Files

- **`logistic_regression_breast_cancer.ipynb`**: Jupyter notebook containing the entire implementation of logistic regression, including data preprocessing, model training, and evaluation.
- **`breast_cancer_data.xlsx`**: Dataset with features related to breast cancer diagnosis.

## Dataset Features

The dataset consists of several features used for predicting whether the tumor is malignant or benign:

1. **Year**: Year of diagnosis.
2. **Age**: Age of the patient.
3. **Menopause**: Menopause status (pre-menopause or post-menopause).
4. **Tumor Size (cm)**: Tumor size in centimeters.
5. **Inv-Nodes**: Number of involved lymph nodes.
6. **Breast**: Affected breast (left or right).
7. **Metastasis**: Presence of metastasis (Yes/No).
8. **Breast Quadrant**: The breast quadrant where the tumor is located.
9. **History**: Family history of breast cancer (Yes/No).
10. **Diagnosis Result**: Binary label indicating malignant (1) or benign (0).

## Model Implementation

### Steps in the Notebook:

1. **Data Loading**: Load the dataset from the Excel file using `pandas`.
2. **Data Preprocessing**:
   - Handle any missing data.
   - Convert categorical features (e.g., Menopause, Breast) into numerical values.
   - Normalize the features for better performance in gradient descent.
3. **Logistic Regression from Scratch**:
   - Implement logistic regression by defining the cost function and gradients.
   - Use gradient descent to optimize model weights.
4. **Model Training**: Train the model on the breast cancer dataset.
5. **Evaluation**: Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.

## Results

After training and evaluating the logistic regression model on test data, the following results were achieved:

- **Accuracy**: 0.97
- **Precision**: 0.96
- **Recall**: 1.00
- **F1 Score**: 0.98

The model demonstrates high accuracy in predicting breast cancer diagnosis, with perfect recall, meaning all malignant tumors were correctly identified.


This high accuracy, precision, recall, and F1-score indicates the model's strong performance in both identifying benign and malignant tumors.

## Requirements

Install the required Python libraries before running the notebook:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Arash_Azhand/Deep-Learning.git
   cd Deep-Learning/Logistic-Regression-Breast-Cancer
   ```
2. Install the required dependencies using `pip`.
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook Logistic-Regression-Breast-Cancer.ipynb
   ```
4. Run all cells in the notebook to load the data, preprocess it, train the model, and evaluate the results.

## Future Improvements

- Apply **regularization** techniques such as L1 or L2 to improve generalization.
- Implement logistic regression using **Scikit-learn** and compare its performance with the scratch implementation.
- Extend the project to handle **multi-class classification**.

## License

This project is licensed under the MIT License.

