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

where 𝑤 are different words in the document. 𝑖
Before training the NB model, we first preprocess the data (original reviews from Yelp). We change all
words in lowercase, remove all punctuations and special characters as well as stop words such as "the",
"a", and "is". Besides, stemming words also plays an important role in our preprocessing. To train and test
the model, we first split the preprocessed dataset into training and testing datasets with an 80-20 split.
There’s no need to reinvent the wheel when it works and is efficient. We also use the CountVectorizer()
class to convert the collection of text documents into a matrix of token counts. Therefore, we use
multinomial Naive Bayes model provided by Scikit-learn to classify the documents with additive
(Laplace/Lidstone) smoothing parameter equal to one. The result of the NB model on real and synthetic
data is presented and discussed in Section 3.





