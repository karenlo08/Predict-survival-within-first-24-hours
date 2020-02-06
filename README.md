# Probability of patient's survival within first 24 hours of intensive care 

## Table of contents
- [Goal]
- [Exploratory Data Analysis]
- [Feature Selection]
- [Machine Learning Models]

## Goal
The challenge is to create a model that uses data from the first 24 hours of intensive care to predict patient survival. MIT's GOSSIS community initiative, with privacy certification from the Harvard Privacy Lab, has provided a dataset of more than 130,000 hospital Intensive Care Unit (ICU) visits from patients, spanning a one-year timeframe. This data is part of a growing global effort and consortium spanning Argentina, Australia, New Zealand, Sri Lanka, Brazil, and more than 200 hospitals in the United States ro develop a new family of open source scoring systems for assessing the severity of illness of critical care patients internationally.

differences in care systems and patient populations may translate the same score to very different outcomes. Put simply, an illness severity translates into different mortality risks depending on where the patient is located. 

They have distinct scoring methods and want to find the best ones to predict survival

Dataset Source: https://gossis.mit.edu/


## Exploratory Data Analysis

## Feature Selection

```python
def normalize(df,column):
    cols_to_norm = [column,'Total_EV']
    df[cols_to_norm] = StandardScaler().fit_transform(df[cols_to_norm])
    return df
```

<img src="/visualizations/.png"/>

##  Machine Learning Models



