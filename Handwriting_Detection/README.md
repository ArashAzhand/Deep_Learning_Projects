# Handwriting Symbol Classification

This project focuses on classifying handwritten symbols using a convolutional neural network (CNN). The dataset contains images of digits (0-9), selected lowercase letters (`w, x, y, z`), and arithmetic symbols (`+, -, /, *`). This project aims to design and train CNN models to classify these symbols accurately.

## Dataset

- **Content**: The dataset contains handwritten images of digits (`0-9`), letters (`w, x, y, z`), and four arithmetic symbols:
  - `+` (plus)
  - `-` (minus)
  - `/` (divide)
  - `*` (multiply, represented by a dot symbol in the dataset)

- **File Organization**: Each image is named with a prefix that represents its label, followed by additional identifier information. 

## Project Workflow

1. **Dataset Preparation**:
   - Extract images from the provided ZIP file.
   - Assign labels based on the prefix of the filenames (digits, letters, or arithmetic symbols).
   - Split the dataset into training, validation, and test sets using stratified sampling to maintain class balance.

2. **Data Visualization**:
   - Display a sample of images along with their labels for a quick visual inspection.
   - Plot the class distribution for training, validation, and test sets to ensure data balance.

![image](https://github.com/user-attachments/assets/95d7e453-e3c2-4380-abb0-7df69bba3e2e)


3. **CNN Model Design**:
   - Two CNN models are designed:
     - **Underfitting Model**: A simpler model with fewer layers to demonstrate underfitting.
     - **Overfitting Model**: A more complex model with additional layers to show overfitting.

4. **Training and Evaluation**:
   - Train both models with different learning rates to observe their impact on model performance.
   - Use **cross-entropy loss** and **Adam optimizer** for training.
   - Evaluate the model on the validation set and record performance metrics like loss and accuracy using **Weights & Biases (wandb)** for experiment tracking.

5. **Final Training and Testing**:
   - Train the best-performing model with the optimal learning rate for 10 epochs.
   - Evaluate the final model on the test set to obtain test accuracy and loss.

## Results

- The **Overfitting Model** achieved the best performance, highlighting the need for model complexity in image classification tasks.
- **Best Learning Rate**: Selected after comparing validation accuracies at different rates.
- **Test Accuracy**: Achieved high accuracy on the test set `0.9815`, demonstrating effective symbol classification.

## Dependencies

- `torch`: For model development and training.
- `torchvision`: For data transformations.
- `PIL`: To handle image files.
- `scikit-learn`: To split the dataset.
- `matplotlib`: For visualizations.
- `wandb`: For experiment tracking.

## How to Run

1. Extract the dataset to the specified path (`/content/data/symbols`) in google colab.
2. Install the required dependencies:
   ```sh
   pip install torch torchvision Pillow scikit-learn matplotlib wandb
   ```
3. Train the model:
   - Modify the script to point to the correct dataset path.
   - Run the script to start the training process.

## Key Takeaways

- **Model Selection**: Balancing model complexity is crucial. The underfitting model lacked sufficient capacity, while the overfitting model performed well but might require regularization techniques to generalize better.
- **Data Augmentation**: Future improvements could include applying data augmentation techniques to enhance the model's robustness.
- **Experiment Tracking**: Tracking different configurations using **wandb** helped identify the optimal learning rate and other hyperparameters.

