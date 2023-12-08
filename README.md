# Image_Classification

MODEL SUMMARY

1. Dataset Loading and Preprocessing:
   - The code loads images of five celebrities (Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, Virat Kohli) from a specified directory (`'Dataset_Celebrities\\cropped'`).
   - The number of images for each celebrity is printed.

2. Model Architecture:
   - The model is a sequential model using Keras with Convolutional Neural Network (CNN) layers.
   - It consists of Conv2D, MaxPooling2D, Flatten, Dense, Dropout, and Dense layers.
   - The model is designed for image classification with an input shape of (128, 128, 3) and an output of 5 classes (softmax activation).

3. Data Splitting and Normalization:
   - The dataset is split into training and testing sets using `train_test_split`.
   - The pixel values of images are normalized by dividing by 255.

4. Model Compilation:
   - The model is compiled using the Adam optimizer, 'CategoricalCrossentropy' loss, and 'accuracy' metric.

5. One-Hot Encoding:
   - Labels are one-hot encoded using `to_categorical`.

6. Model Training:
   - The model is trained using the training data for 50 epochs with a batch size of 128.
   - Training history is stored in the variable `history`.

7. Model Evaluation:
   - The trained model is evaluated on the test set.
   - Accuracy and a classification report (precision, recall, f1-score) are printed.

8. Model Prediction:
   - The code includes a section for predicting classes for new images using the trained model.
   - It provides predictions for sample images of each celebrity.

CRITICAL FINDINGS
- The code uses a relatively simple CNN architecture, which may not capture complex features in the images effectively.
- The choice of hyperparameters, such as the number of layers, filter sizes, and dropout rates, may impact model performance. Tuning these parameters could be beneficial.
- The absence of data augmentation may limit the model's ability to generalize well to unseen data. Uncommenting the data augmentation section and experimenting with augmentation parameters could improve performance.
- The use of a more sophisticated pre-trained model or transfer learning could be considered for better feature extraction, especially if the dataset is not large.
- The model evaluation includes accuracy and a classification report, providing a detailed breakdown of performance metrics for each class.
