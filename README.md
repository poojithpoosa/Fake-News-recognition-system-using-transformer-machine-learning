# Fake-News-recognition-system-using-transformer-machine-learning

Today, millions of individuals are always connected online, sharing and receiving data that may be useful or harmful, or that could mislead them in the wrong direction, all because they can't tear themselves away from their smartphones and other digital devices. The term "fake news" refers to stories that intentionally mislead its readers or include fabricated facts in an effort to influence their opinions. It is possible for fake information to become viral on the internet, impacting public opinion and policy much like reliable information does. My goal with this work is to develop an automated method for detecting and countering false news stories. The Lair dataset was utilised, which has roughly 12.8 thousand news statements compiled over a decade from several reliable sources. I have developed a model of the switch-transformer with uses self-additive attention. The suggested model achieved an accuracy of 99.52% on the test set and 99.77% on the training set, with a loss of 0.012 on the test set and 0.009 on the training set, respectively, demonstrating no overfitting and steady training.
## AIM

The project's overarching goal is to use switch transformers using self-additive attention in the identification of Fake news. When compared to other sequence-to-sequence models, Transformer is an architecture consisting of two components (Encoder and Decoder) that performs the transformation. My project will make use of the Lair dataset, and the model will be tested for accuracy, loss error.

## Research questions
*	What are the performance gains when using switch transformers?
*	Does using self-additive attentions have any effect on model performance?

# Various step in the project:
1.	Dataset loading: The Lair dataset is loaded into the python environment.
2.	Dataset exploration: The dataset is explored, looking for balance in data, errors and developing graphs.
3.	Dataset preprocessing: The dataset is pre-processed using various preprocessing techniques.
4.	Creating model: The proposed model which is Switch-Bert transformer using self-adaptive attention is created using python libraries.
5.	Training the model: The model is trained on pre-processed dataset and tested on test set.

Proposed model : 

![image](https://user-images.githubusercontent.com/61981756/199019514-2f0f5523-6b24-49e4-9dec-a17d58e47775.png)

Results:

![image](https://user-images.githubusercontent.com/61981756/199019738-17a3cce8-4d64-46a6-a516-e1dfc9b3f70e.png)

![image](https://user-images.githubusercontent.com/61981756/199019786-2633930a-fa34-45a3-8ed7-a032a2a6f9cc.png)
