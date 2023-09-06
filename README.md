# Sentiment_Analysis_on_Yelp_Reviews

[![Python application test with Github Actions](https://github.com/nogibjj/YZ_NLP/actions/workflows/main.yml/badge.svg)](https://github.com/nogibjj/YZ_NLP/actions/workflows/main.yml)

## 1. Introduction
In this project, we are interested in the topic of sentiment analysis, which is an approach in natural
language processing that identifies the emotional tone behind a body of text. In practical application,
sentiment analysis tools are essential to detect and understand customer feelings. It is an important factor
when it comes to product and brand recognition, customer loyalty, customer satisfaction, advertising and
promotion's success, and product acceptance. Hence, we chose the dataset from Kaggle, which consists of
reviews from Yelp in 2015, and we will apply Naïve Bayes model and LSTM (Long Short-Term Memory)
on this data and analyze the results.

## 2. Methodology
Our Yelp reviews dataset was obtained from Kaggle and is made up of Yelp reviews [1]. It was taken
from the 2015 Yelp Dataset Challenge data information. The Yelp review polarity dataset is created by
taking into account the analysis of positive and negative reviews. There are 38,000 testing samples and
560,000 training samples combined in total in Kaggle. Class 1 polarity is negative, and class 2 polarity is
positive.
Because the train.csv is large, we mainly use the test.csv file to analyze Yelp reviews and found that it’s
large enough for training models. It contains two columns, which refers to class index (1 and 2) and
review text. Double quotes (") are used to escape the review paragraphs, and any inside double quotes are
separated by two double quotes (""). A backslash and a "n" letter, which together form the symbol "\n,"
are used to separate new lines.

### 2.2 Generative model – Naive Bayes
Naïve Bayes [2] (NB) is one of the simplest generative probabilistic models to solve the classification
problem. The foundation behind NB is the Bayes’ rule:

<img width="159" alt="Screen Shot 2023-09-06 at 5 44 57 PM" src="https://github.com/helenyjx/NLP-Final/assets/112274822/e987391f-e909-4dcb-aa63-e5c87b9a527c">

where is the conditional probability of an event happening given that another 𝑃(𝑦) 𝑥 event 𝑦 has already
occurred. In our problem, we would like to classify Yelp reviews as negative or positive ones, labeled as 1
and 2 respectively. Then how should we treat or encode the text? A method called bag-of-words (BoW) is
adopted. A BoW is a representation of text that describes the occurrence of words within a document,
ignoring their order. For example, a piece of reviews reads:
“The food is good. Unfortunately, the service is very hit or miss. The main issue seems to be with the
kitchen, the waiters and waitresses are often very apologetic for the long waits and it is pretty obvious
that some of them avoid the tables after taking the initial order to avoid hearing complaints”.
Immediately, we notice that the occurs 8 times, and occurs 2 times, very occurs 2 times and so on. The
BoW looks like:

<img width="735" alt="Screen Shot 2023-09-06 at 5 45 43 PM" src="https://github.com/helenyjx/NLP-Final/assets/112274822/c337a5b9-148b-479f-8889-c00772581965">

Obviously, some words are useless, such as the, is, and and. On the other hand, the word avoid appears
twice and unfortunately shows up once. We may consider it as a negative review. In fact, it is. After
preprocessing the raw reviews, most of meaningless words (stopwords) will be removed from documents,
and only useful words will be kept. For a document 𝑑, our best estimate of the correct class 𝑐 is:

<img width="215" alt="Screen Shot 2023-09-06 at 5 46 50 PM" src="https://github.com/helenyjx/NLP-Final/assets/112274822/8935d898-b8c6-4305-b8d8-7a3d910ffe6e">

By the Bayes’ rule and the assumption of independency of features of each documents, we can have:

<img width="441" alt="Screen Shot 2023-09-06 at 5 47 18 PM" src="https://github.com/helenyjx/NLP-Final/assets/112274822/5db1b0c2-335d-478b-8d22-8073bfd91295">

where 𝑤𝑖 are different words in the document. 

Before training the NB model, we first preprocess the data (original reviews from Yelp). We change all
words in lowercase, remove all punctuations and special characters as well as stop words such as "the",
"a", and "is". Besides, stemming words also plays an important role in our preprocessing. To train and test
the model, we first split the preprocessed dataset into training and testing datasets with an 80-20 split.
There’s no need to reinvent the wheel when it works and is efficient. We also use the CountVectorizer()
class to convert the collection of text documents into a matrix of token counts. Therefore, we use
multinomial Naive Bayes model provided by Scikit-learn to classify the documents with additive
(Laplace/Lidstone) smoothing parameter equal to one. The result of the NB model on real and synthetic
data is presented and discussed in Section 3.

### 2.3 Discriminative model – LSTM [3]
Our neural network model is based on LSTM which is an efficient way to do sentiment analysis since it is
able to remember and utilize information from long-term dependencies in the input data. We use
Tensorflow to realize this purpose. For the neural network, the input needs to be numeric. We first
initialize the tokenizer with a 5000 word limit, which is the number of words we would like to encode.
Then we create associations of words and numbers, and replace the words in a sentence with their
respective associated numbers. Since the length of each document is different and LSTM requires inputs
to have equal lengths, we have to add the sequence to have the chosen length of inputs (20 in our model).
The structure and parameters of LSTM is as shown below.

<img width="532" alt="Screen Shot 2023-09-06 at 5 48 47 PM" src="https://github.com/helenyjx/NLP-Final/assets/112274822/f299692d-e418-4698-b2a8-a87bc9d63793">

Our neural network contains 5 layers. The first layer is an embedding layer, which converts the input
sequences into dense vectors of a fixed size, called embedding vectors. The length of the embedding
vector is specified as 32 and input length is 20. The second layer is a spatial dropout layer, which
randomly sets 25% of input units to 0 at each update during training time to help prevent overfitting. The
third layer is an LSTM layer with 50 units, which is a type of recurrent neural network that can process
long sequences of data. The dropout percent and recurrent drop-out percent are both set to be 50. The
fourth layer is a dropout layer, which randomly sets 20 percent of input units to 0 at each update. The final
layer is a dense layer with sigmoid function as activation function. Since our problem is a binary
classification problem, binary cross entropy is suitable as loss function and Adam is used as an optimizer.
To train the model, we found that 10 epochs and a batch size of 32 are enough to generate a stable result.
The result of the LSTM model on real and synthetic data is presented and discussed in Section 3 as well.

### 2.4 Synthetic data generation [4]
One reason for calling Naïve Bayes a generative model is that NB can generate synthetic data. When we
have a NB model, it means that we have estimates of <img width="125" alt="Screen Shot 2023-09-06 at 5 50 18 PM" src="https://github.com/helenyjx/NLP-Final/assets/112274822/3506159b-cb44-43b9-82ab-a5ef2dde55db"> , which 𝑃(𝑐) we can use to generate documents. We have to generate sentences word by word. First, we sample a class from 𝑃(𝑐). Next, we
keep generating words by sampling from <img width="147" alt="Screen Shot 2023-09-06 at 5 52 02 PM" src="https://github.com/helenyjx/NLP-Final/assets/112274822/ec1f2703-14ad-4fbe-bab8-066c86e41701">




