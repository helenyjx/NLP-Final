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


