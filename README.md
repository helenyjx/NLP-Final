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
