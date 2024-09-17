# SMS Spam Classifier

An SMS Spam Classifier that uses Natural Language Processing (NLP) and machine learning to classify SMS messages as ham (not spam) or spam. The model achieves high accuracy and precision, making it effective for detecting unwanted messages.

## Dataset


The dataset used for this project was sourced from Kaggle. It consists of SMS messages labeled as either ham or spam.
Link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Data Preprocessing
Column Selection: Removed unnamed columns and retained only the target column (ham or spam) and the SMS message column.

Label Encoding: Converted the target labels into numerical format using LabelEncoder, where:

ham = 0
spam =1

## Exploratory Data Analysis

#### Pandas Profiling:
 Conducted an initial data analysis to understand the dataset's structure and content.

#### Feature Engineering: 
Added new columns to extract additional information:

#### num_characters: 
Number of characters in the message.

#### num_words: 
Number of words in the message.

#### num_sentences: 
Number of sentences in the message.

#### Correlation Analysis:
 Plotted a correlation matrix to identify relationships between features.

### Feature Selection: 
Due to high correlation among the new features, only num_characters was retained for further analysis.

## Text Pre-Processing

Applied NLP techniques to preprocess the text data:

Lowercasing: Converted all text to lowercase.

Tokenization: Split text into individual words.

Removing Special Characters: Eliminated punctuation and special symbols.

Removing Stop Words and Punctuation: Removed common words that do not contribute to the meaning (e.g., 'the', 'and') and punctuation marks.

Stemming: Reduced words to their root form using stemming algorithms.


## Model Building

Converted the preprocessed text into numerical features suitable for machine learning algorithms using techniques like TF-IDF Vectorization.

Multiple algorithms were tested to identify the most effective model:

1) Naive Bayes (NB)

2) Support Vector Classifier (SVC)

3) Random Forest (RF)

4) Logistic Regression (LR)

5) K-Nearest Neighbors (KNN)

6) Decision Tree (DT)

7) AdaBoost
Gradient Boosting (GBDT)

8) Extreme Gradient Boosting (XGB)

9) Bagging Classifier (BgC)

10) Extra Trees Classifier (ETC)


## Model Evaluation

Evaluated the performance of each model using Accuracy and Precision metrics:

```bash
| Algorithm                         | Accuracy  | Precision |
| ---------------------------------- | ---------:| ---------:|

| Multinomial Naive Bayes (NB)       | 0.970986  | 1.000000  |

| Random Forest (RF)                 | 0.975822  | 0.982906  |

| Support Vector Classifier (SVC)    | 0.975822  | 0.974790  |

| Extra Trees Classifier (ETC)       | 0.974855  | 0.974576  |

| Logistic Regression (LR)           | 0.958414  | 0.970297  |

| AdaBoost                           | 0.960348  | 0.929204  |

| Extreme Gradient Boosting (XGB)    | 0.967118  | 0.926230  |

| Gradient Boosting (GBDT)           | 0.946809  | 0.919192  |

| Bagging Classifier (BgC)           | 0.958414  | 0.868217  |

| K-Nearest Neighbors (KNN)          | 0.905222  | 1.000000  |

| Decision Tree (DT)                 | 0.929400  | 0.828283  |
```

1) Best Model: The Multinomial Naive Bayes classifier achieved the highest precision and excellent accuracy.

2) Ensemble Methods: Tried ensemble methods like Random Forest and 
AdaBoost, but none surpassed the performance of Naive Bayes.


## Deployment

Deployed the final model using Streamlit, creating an interactive web application where users can input an SMS message and receive a prediction.

