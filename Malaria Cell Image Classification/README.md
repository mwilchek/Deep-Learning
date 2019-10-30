# Malaria Cell Image Classification with MLP

## Introduction
This exercise consists in using a Multi-Layer Perceptron (MLP) to classify blood cells into 4different categories: “red blood cell”, “ring”, “schizont” and “trophozoite”. Apart from the first type, the rest of the cells indicate some stage of malaria infection.  You can read more about this disease in the following link:https://www.cdc.gov/malaria/about/biology/.  If a model can successfully identify these types of cells from images taken with a microscope, thiswould allow to automate a very time-consuming testing process, leaving human doctors withmore time to treat the actual disease. Furthermore, an early malaria detection can save lives!

The format will be a competition between you and the rest of your classmates.  You will be given a training set and then asked to submit your code together with your trained MLP, sothat we can test it on our private held-out set. You will be given a unique private nickname tocheck your results in the public leaderboard, and you may submit your work once per day until the competition ends. The leaderboard will be updated once everyday during the morning, andwhen the competition ends, the final results will be published with the students’ real names.

As you will find out, this dataset is very imbalanced, so the accuracy is not the best metricto use for the leaderboard. Instead, we will use the macro-averaged F1-score and the Cohen’s Kappa score, and the ranking in the leaderboard will be determined by the mean of these twoscores. The functions used to obtain them are provided in the sample codes (explained below).You will have a Daily Board and Leader Board excel sheet.  The Daily Board shows your progress every day however the leader board show your ranking based on the best score out of all the days. These sheets will be updated every day in electronic reserve section.

## Data set and sample code
- Download the exam dataset from the following link https://storage.googleapis.com/exam-deep-learning/train.zip.
- Please check the sample codes first and download them from https://github.com/amir-jafari/Deep-Learning/tree/master/Exam_MiniProjects/3-Keras_Exam1_Sample_Codes_F19 and then you can write your own code. This is just get you started and give you an example of how you write your predict function and save your models.

## Rules of the Competition
- You  can only use  MLP  for  training.   This  means  only  Fully  Connected  layers. No Convolutional Layers nor feedbacks are allowed (LSTMor any RNN architecture). No pre-trained models are allowed such as (resnet, densenet, ...)
- You can use other operations in-between layers, like Dropout, Batch Normalization or other types of layer input/output manipulation.
- You can only use the data you are given. Using additional data from any other sources is not allowed.
- You can do any kind of pre-processing with the training data, which you should split intoat least training and testing. You may use whichever library you want for this purpose.
- You can only use Keras for training the model.
- You are allowed to search in the internet and find out ideas.  You can use any external GitHub but you need to cite it.  If found any violation of this rule you get a reduced grade.

