# Ciphar-10-Project
This project builds a complex Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. It leverages techniques like data augmentation, BatchNormalization, Dropout, and early stopping to enhance model performance and prevent overfitting. The model is trained using the Keras API, with a focus on optimizing accuracy through learning rate scheduling and effective dropout strategies.

Key Features:

CNN Architecture: Stacked convolutional layers with increasing filter depth and regularization via dropout.

Data Augmentation: Applied random rotations, shifts, flips, and zooms to the training images to improve generalization.

Early Stopping & LR Scheduling: Integrated callbacks to monitor and reduce overfitting.

Visualization: Plots of training/validation accuracy and a display of misclassified images for deeper insight into model performance.

Tools: TensorFlow, Keras, Matplotlib, Seaborn
