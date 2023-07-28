# Diabetic-Retinopathy-Detection-and-stage-classification-using-Lenet-5
Detect and classify diabetic retinopathy stages using the LeNet-5 architecture. Leverage retinal images with labeled severity levels to build an accurate model for early detection and treatment of diabetic retinopathy.


Overview
This project aims to develop a system for detecting diabetic retinopathy in retinal images and classifying its severity into different stages using the LeNet-5 architecture. Diabetic retinopathy is a serious eye condition that affects people with diabetes, and early detection is crucial for timely treatment and prevention of vision loss.

Dataset
The dataset used for this project consists of retinal images captured from patients diagnosed with diabetic retinopathy. Each image is labeled with the corresponding stage of diabetic retinopathy, ranging from 0 (no diabetic retinopathy) to 4 (proliferative diabetic retinopathy). The dataset is divided into training, validation, and testing sets to train and evaluate the model.

LeNet-5 Architecture
LeNet-5 is a classic convolutional neural network (CNN) architecture proposed by Yann LeCun in 1998. It consists of several layers, including convolutional layers, pooling layers, and fully connected layers. The LeNet-5 architecture is well-suited for image classification tasks, making it a suitable choice for our diabetic retinopathy detection system.

Dependencies
The following libraries are required to run the project:

Python (>= 3.6)
TensorFlow (>= 2.0)
Keras (>= 2.4)
NumPy (>= 1.19)
OpenCV (>= 4.5)
Matplotlib (>= 3.3)

Results
The trained LeNet-5 model achieved an accuracy of 97% on the test set and can effectively classify the severity of diabetic retinopathy.
