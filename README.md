
# Hate Speech detection Web-App 


This is a group project for Continuous Assesment at DIEMS. We have built an interface that will classify a comment or statement as toxic, severe_toxic, obscene, threat, insult or identity_hate. Here, we have used 'Gradio' to create the interface and the Classification Models has been created on Google Colab. The Dataset we have used is available on Kaggle - https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge



## Model Description

**Dataset Overview**

The threat of abuse and harassment online prevent many people from expressing themselves and make them give up on seeking different opinions. In the meantime, platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments. Therefore, Kaggle started this competition with the Conversation AI team, a research initiative founded by Jigsaw and Google. The competition could be found here: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

As a group of students with great interests in Natural Language Processing, as well as making online discussion more productive and respectful, we determined to work on this project and aim to build a model that is capable of detecting different types of toxicity like threats, obsenity, insults, and identity-based hate.

The dataset we are using consists of comments from Wikipedia’s talk page edits. These comments have been labeled by human raters for toxic behavior. The types of toxicity are:

toxic
severe_toxic
obscene
threat
insult
identity_hate
There are 159,571 observations in the training dataset and 153,164 observations in the testing dataset. Since the data was originally used for a Kaggle competition, in the test_labels dataset there are observations with labels of value -1 indicating it was not used for scoring.


**Understanding the Data**

The first step is understanding the data that we are going to use. So, we have performed various steps for the same:

1) **Data Preprocessing and EDA :**  The major concern of the data is that most of the comments are clean (i.e., non-toxic). There are only a few observations in the training data for Labels like threat. This indicates that we need to deal with imbalanced classes later on and indeed, we use different methods, such as resampling, choosing appropriate evaluation metrics, and choosing robust models to address this problem.

2) **Model Fitting :** During the modeling process, we choose multiple different evaluation metrics to evaluate the performance of models based on the nature of our data:

i)Recall

ii)F Score

iii)Hamming Loss

Using Multinomial Naive Bayes as our baseline model, we first used k-fold cross validation and compared the performance of the following three models without any hyperparameter tuning: Multinomial Naive Bayes, Logistic Regression, and Linear SVC. Logistic Regression and Linear SVC perform better than Multinomial Naive Bayes.

After checking how these models perform on the test data, we notice that Muninomial Naive Bayes does not perform as well as the other two models while Linear SVC in general out performs the others based on F1 score.

After accounting for the imbalanced data, the F1 score of Logistic Regression model has jumped to an average of 0.9479 while Linear SVC has jumped to 0.9515.

With the help of **grid search**, we were able to find the "optimal" hyperparameters for the models and have reached an average of the best score of 0.9566 for Logistic Regression and 0.9585 for Linear SVC.




## DeepLearning Approach


After performing preprocessing and vectorization,
we create a sequebtial model to create a Deep Neural network. 

We have used LSTM (i.e., Long Short Term Memory)
along with Dropout, Bidirectional and Dense Layers from tensorflow.keras.Layers

The LSTM layer has 32 different LSTM units i.e., neurons and has tanh activation function.
*We are using tanh instead of Relu here, because, the GPU acceleration that is required for an LSTM layer needs to be tanh.*

Bidirectional is used in RNN to pass information in both sides of the neural network. Therefore, it is important in the case of NLP.

*Dense layers perform the job of feature extaction*

**Evaluation Metrics**

1) Precision = 0.81080162525 
2) Recall = 0.683366715908085
3) CategoricalAccuracy = 0.46740221977233887

*We are woring on improving the model's accuracy*



## Dependencies

All you need is Google Colab and the required dataset to create this WebApp on your device
