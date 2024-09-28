# Student Score Prediction using Neural Networks

## Overview

This project aims to predict student performance by using a **neural network** built with **TensorFlow**. The model takes into consideration multiple factors that can affect student scores, such as study habits, previous performance, and lifestyle factors. The goal is to accurately predict student scores and gain insight into the major contributors to their academic success.

## Project Files

- **`NeuralNetwork_Score_Prediction.ipynb`**: The Jupyter notebook containing the complete implementation, including data preprocessing, neural network modeling, and evaluation.
- **`Q2.csv`**: The dataset used for training and testing the model, containing relevant features related to student performance.

## Dataset Features

The dataset includes the following features that were used to predict student performance:

1. **Hours Studied**: Number of hours each student spends studying.
2. **Previous Scores**: Scores from previous assessments or exams.
3. **Extracurricular Activities**: A score representing student participation in extracurricular activities.
4. **Sleep Hours**: Average number of hours of sleep per night.
5. **Sample Question Papers Practiced**: Number of sample question papers practiced by the student.
6. **Performance Index**: Overall performance index calculated based on various criteria.

## Model Implementation

### Steps in the Notebook:

1. **Data Loading**: Load the dataset from a CSV file using `pandas`.
2. **Data Preprocessing**:
   - Handle missing values if present.
   - Normalize the features to improve model training efficiency.
   - Split the dataset into training and testing sets.
3. **Neural Network Creation**:
   - Use **TensorFlow** to create a feedforward neural network model.
   - Define the model architecture, including input, hidden, and output layers.
4. **Model Training**:
   - Compile the model using an appropriate loss function and optimizer (e.g., **Mean Squared Error (MSE)** loss and **Adam** optimizer).
   - Train the model on the training data.
5. **Model Evaluation**:
   - Evaluate the model using metrics such as **Mean Absolute Error (MAE)** and **R-squared (R²)** on the test data.
   - Visualize training and validation losses over epochs.

## Results

The model achieved the following performance on the test set:

- **R-squared (R²)**: **0.989**  
  This means that approximately **98.9% of the variance** in student scores can be explained by the features included in the model, indicating a strong fit and accurate prediction.

- **Mean Absolute Error (MAE)**: Indicates the average difference between predicted and actual values, showing how close the predictions are to the true student scores.

### Visual Representation

To provide a better understanding of the model's predictions:

- A **scatter plot** is used to visualize the actual versus predicted scores for individual students.
- A subset of data points is plotted to make the comparison clearer and to better assess model performance.

![image](https://github.com/user-attachments/assets/6a82d9d3-806b-4ed2-817d-c302113604ad)


## Requirements

Install the required Python libraries before running the notebook:

```bash
pip install numpy pandas tensorflow matplotlib scikit-learn
```

## Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/ArashAzhand/Deep_Learning_Projects/Neural_Networks_Student_Score_Prediction.git
   cd student-score-prediction
   ```
2. Install the required dependencies using `pip`.
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook NeuralNetwork_Score_Prediction.ipynb
   ```
4. Run all cells to see the neural network's training process and evaluate its performance.


## Future Improvements

- Experiment with different **neural network architectures**, such as adding more hidden layers or changing the number of neurons.
- Use **regularization** techniques to avoid overfitting and improve generalization.
- Experiment with different activation functions and optimizers to find the best combination for this dataset.

