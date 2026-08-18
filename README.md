Bank Marketing – Machine Learning Classification
1. Problem Statement
The objective of this project is to build and compare multiple machine learning classification models for predicting whether a **bank customer** will subscribe to a term deposit as a result of a direct marketing campaign.
The main goal of this project is to train different machine learning algorithms and compare their performance using the following evaluation metrics:

2. Dataset Description
The dataset is obtained from the UCI Machine Learning Repository – Bank Marketing Dataset [https://archive.ics.uci.edu/dataset/222/bank+marketing]. The original data represents direct marketing campaigns conducted by a Portuguese banking institution through phone calls.
The target variable is y, which has two possible values:
**yes** – the customer subscribed to a term deposit.
**no** – the customer did not subscribe to a term deposit.

3. GitHub Repository Link: https://github.com/RakeshPaine/psychic-spoon.git

4. Model's Metrics:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9102 | 0.9317 | 0.6661 | 0.4182 | 0.5138 | 0.4825 |
| Decision Tree | 0.9077 | 0.8747 | 0.6036 | 0.5455 | 0.5730 | 0.5223 |
| kNN | 0.8948 | 0.8595 | 0.5515 | 0.3893 | 0.4564 | 0.4075 |
| Naive Bayes | 0.8506 | 0.8498 | 0.3976 | 0.6150 | 0.4830 | 0.4133 |
| Random Forest | 0.9133 | 0.9433 | 0.6495 | 0.5134 | 0.5735 | 0.5305 |

5. Observations on the performance of each model.

| ML Model Name | Observation about model performance |
| :---: | :---: |
| Logistic Regression| Strongest baseline precision (66.6%) and excellent AUC, but struggles heavily with high false negatives |
| Decision Tree | Well-balanced model with decent Precision and Recall, yielding a strong F1-score, but falls slightly short of Random Forest. |
| kNN | Poor overall performer with weak Recall (38.9%) and the lowest F1-score/MCC among the top models. |
| Naive Bayes | Lowest accuracy and precision, but achieves the highest Recall (61.5%), making it the best at capturing positive cases. |
| Random Forest | Best overall performance. Top-tier Accuracy, highest AUC (94.33%), highest F1-score (57.35%), and highest MCC. |
| Overall Winner? | **Random Forest.** It dominates in almost every metric, showing the most robust and reliable predictive power. |
