# OCR-tech-by-ML
## Introduction
This is a simple OCR program use machine learning to train the natural network to recognize 26 capital character. With the improvement of the train data, the accuracy of the program will continuously increase.
## Theory
The program converts the input image into a grayscale image and then converts it into a 289*1 matrix (because the training data is in a 17*17 pixel format). Program Randomly selects points to obtain the initial calculation parameters, and continuously adjust the parameters through forward and backward propagation to improve the accuracy. 
## Demo
This is a simple B with 100*100 pixel test
![image](https://github.com/cjx5021/OCR-tech-by-ML/blob/main/Demo/Demo.png)
This is a blue A with 800*800 pixel test

## Paper
