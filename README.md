# Prediction of patient survival within first 24 hours of intensive care 

## Table of contents
- [Goal](#general-info)
- [Workflow](#general-info)
- [Feature Selection](#prediction-model)
- [Machine Learning Models](#exploratory-data-analysis)
- [Lessons learned](#exploratory-data-analysis)


## Goal
Create a model that uses data from the first 24 hours of intensive care to predict patient survival. GOSSIS community initiative has provided a dataset of more than 130,000 hospital Intensive Care Unit (ICU) visits from patients. This data is part of a growing global effort and consortium spanning Argentina, Australia, New Zealand, Sri Lanka, Brazil, and more than 200 hospitals in the United States to develop a new family of open source scoring systems for assessing the severity of illness of critical care patients internationally.

Dataset Source: https://gossis.mit.edu/

## Workflow
<img src="/img/Screen Shot 2020-02-07 at 1.51.07 PM.png"/>

## Why Feature Selection?
Although regularization helps constrain or shrinks our coefficient towards zero and help us to avoid overfitting;
Most of the times, we will have many non-informative features where poor-quality input will produce poor-quality output.

### Pearson Correlation
Between the target and numerical features in our dataset. 
categorical features should be encoding to 0/1.
```['d1_calcium_min', 'h1_diasbp_min', 'h1_inr_min', 'd1_inr_min', 'h1_diasbp_noninvasive_min', 'd1_wbc_max', 'd1_mbp_invasive_min', 'h1_lactate_max', 'd1_albumin_max', 'ph_apache', 'h1_lactate_min', 'fio2_apache', 'd1_sysbp_invasive_min', 'h1_mbp_min', 'albumin_apache', 'd1_inr_max', 'h1_inr_max', 'h1_mbp_noninvasive_min', 'h1_sysbp_min', 'h1_sysbp_noninvasive_min', 'temp_apache', 'd1_albumin_min', 'd1_hco3_min', 'd1_heartrate_max', 'bun_apache', 'intubated_apache_0.0', 'intubated_apache_1.0', 'd1_bun_min', 'd1_arterial_ph_min', 'd1_bun_max', 'd1_diasbp_noninvasive_min', 'd1_diasbp_min', 'gcs_eyes_apache_4', 'd1_mbp_noninvasive_min', 'd1_mbp_min', 'd1_temp_min', 'gcs_verbal_apache_5', 'd1_spo2_min', 'd1_sysbp_noninvasive_min', 'd1_sysbp_min', 'gcs_verbal_apache_1', 'ventilated_apache_1.0', 'ventilated_apache_0.0', 'gcs_motor_apache_6', 'gcs_motor_apache_1', 'gcs_eyes_apache_1', 'd1_lactate_max', 'apache_4a_icu_death_prob', 'd1_lactate_min', 'apache_4a_hospital_death_prob']
```
<img src="/img/corr.png"/>

For Example:
bun_apache: common blood test, the blood urea nitrogen (BUN) test reveals important information about how well your kidneys and liver are working
intubated_apache	None	binary	Whether the patient was intubated at the time of the highest scoring arterial blood gas used in the oxygenation score


### What is apache II and apache III?
Several scoring systems have been developed to grade the severity of illness in critically ill patients. These systems are moderately accurate in predicting individual survival.


### Chi-Squared
Higher the Chi-Square value the feature is more dependent on the response and it can be selected for model training. Data should be normalize.
sklearn.feature_selection.SelectKBest with chi2 as parameter.

```From sklearn Documentation:
The goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.
```

### Recursive Feature Elimination (RFE)
From sklearn Documentation:
The goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.


### Random Forest
We can also use RandomForest to select features based on feature importance. We calculate feature importance using node impurities in each decision tree. In Random forest, the final feature importance is the average of all decision tree feature importance.


##  Machine Learning Models


## Next steps
Keep going with feature engineering and Gradient Boosting to make a better prediction.
