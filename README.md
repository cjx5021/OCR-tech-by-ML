# OCR-tech-by-ML
## Introduction
This is a simple OCR program use machine learning to train the natural network to recognize 26 capital character. With the improvement of the train data, the accuracy of the program will continuously increase.
## Theory
The program converts the input image into a grayscale image and then converts it into a 289*1 matrix (because the training data is in a 17 * 17 pixel format). Program Randomly selects points to obtain the initial calculation parameters, and continuously adjust the parameters through forward and backward propagation to improve the accuracy. Here we include the following different layers: fully connected layer, activation function layer, loss function layer, data processing layer and accuracy layer.<br/><br/>
Fully connected layer: Initialize parameters, propagate calculation results through forward propagation, and modify parameters through back propagation.<br/><br/>
Activation layer: Add nonlinear factors to the neural network and convert the calculation results into a percentage format which is easy to identify.<br/><br/>
Loss function layer: Calculate the accuracy of the current training data, the starting point of back propagation, and back propagation by calculating the error with the expected input value.<br/><br/>
Accuracy laye: Calculate the accuracy of the program, generally use data different from the training data.
## Demo
This is a simple B with 100 * 100 pixel test
![image](https://github.com/cjx5021/OCR-tech-by-ML/blob/main/Demo/Demo.png)
This is a blue A with 800 * 800 pixel test
![image](https://github.com/cjx5021/OCR-tech-by-ML/blob/main/Demo/Demo_blueA.png)
## Paper
Here is the detailed introduction of this project:
https://github.com/cjx5021/OCR-tech-by-ML/blob/main/machine%20learning%20paper.docx
## Source File
train data txt form: https://github.com/cjx5021/OCR-tech-by-ML/tree/main/train%20data
Under this folder is the train data, test data and validate data's txt form, first colum is the picture's name and second colum is expect result.
main.npy is trained program which will load weight1&2, bias1&2 with 99.3 accuracy.
train.npy is the code to train the model.
## Limitation
Due to the limitation of training data, the recognition of hand-drawn images is not very high. Since the train picture is almost same form with some different interference point. If the hand-draw picture is far with the graph give by the train data, the result will not be correct. This problem can be solved by increasing test data.
