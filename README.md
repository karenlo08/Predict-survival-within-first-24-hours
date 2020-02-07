# Probability of patient's survival within first 24 hours of intensive care 

## Table of contents
- [Goal]
- [Exploratory Data Analysis]
- [Data Preprocessing]
- [Feature Selection]
- [Machine Learning Models]

- [Goal](#general-info)
- [Exploratory Data Analysis](#technologies)
- [Data Preprocessing](#hypotesis-testing)
- [Feature Selection](#prediction-model)
- [Machine Learning Models](#exploratory-data-analysis)


## Goal
The challenge is to create a model that uses data from the first 24 hours of intensive care to predict patient survival. GOSSIS community initiative, with privacy certification from the Harvard Privacy Lab, has provided a dataset of more than 130,000 hospital Intensive Care Unit (ICU) visits from patients, spanning a one-year timeframe. This data is part of a growing global effort and consortium spanning Argentina, Australia, New Zealand, Sri Lanka, Brazil, and more than 200 hospitals in the United States ro develop a new family of open source scoring systems for assessing the severity of illness of critical care patients internationally.

differences in care systems and patient populations may translate the same score to very different outcomes. Put simply, an illness severity translates into different mortality risks depending on where the patient is located. 


Dataset Source: https://gossis.mit.edu/


## Exploratory Data Analysis

## Data Preprocessing
Identifying categorical and numerical features. Treating NaN with mean and one hot encoding.


BMI = formula

Ages with median?




## Feature Selection
Why?

1.If we have more columns in the data than the number of rows, we will be able to fit our training data perfectly, but that won’t generalize to the new samples. And thus we learn absolutely nothing.

2. Most of the times, we will have many non-informative features. For Example, Name or ID variables. Poor-quality input will produce Poor-Quality output.

### Pearson Correlation
Between the target and numerical features in our dataset. 
categorical features should be encoding to 0/1.

<img src="/img/corr.png"/>

bun_apache: common blood test, the blood urea nitrogen (BUN) test reveals important information about how well your kidneys and liver are working

intubated_apache	None	binary	Whether the patient was intubated at the time of the highest scoring arterial blood gas used in the oxygenation score


what is apache II

Several scoring systems have been developed to grade the severity of illness in critically ill patients. These systems are moderately accurate in predicting individual survival. However, these systems are more valuable for monitoring quality of care and for conducting research studies because they allow comparison of outcomes among groups of critically ill patients with similar illness severity.




### Chi-Squared
higher the Chi-Square value the feature is more dependent on the response and it can be selected for model training.


normalization of data

sklearn.feature_selection.SelectKBest with chi2 as parameter.

categorical response (y) and categorical predictors (x)
The SelectKBest function with the chi2 test only works with categorical data. In fact, the result from the test only will have real meaning if the feature only has 1's and 0's.

continuous response (y) and categorical predictor (x)
f_classif (Analysis of variance/ ANOVA)

### Recursive Feature Elimination (RFE)

```From sklearn Documentation:

The goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.
```

### Random Forest

We can also use RandomForest to select features based on feature importance.

We calculate feature importance using node impurities in each decision tree. In Random forest, the final feature importance is the average of all decision tree feature importance.


##  Machine Learning Models


```python
def normalize(df,column):
    cols_to_norm = [column,'Total_EV']
    df[cols_to_norm] = StandardScaler().fit_transform(df[cols_to_norm])
    return df
```

