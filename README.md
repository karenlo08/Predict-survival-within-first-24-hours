# Prediction of patient survival within first 24 hours of intensive care 

## Table of contents
- [Goal](#general-info)
- [Workflow](#general-info)
- [Feature Selection](#prediction-model)
- [Machine Learning Models and Results](#exploratory-data-analysis)
- [Lessons learned and Next Steps](#exploratory-data-analysis)

<img src="/img/gossis_map.png"/>

## Goal
Create a model that uses data from the first 24 hours of intensive care to predict patient survival. GOSSIS community initiative has provided a dataset of more than 130,000 hospital Intensive Care Unit (ICU) visits from patients. This data is part of a growing global effort and consortium spanning Argentina, Australia, New Zealand, Sri Lanka, Brazil, and more than 200 hospitals in the United States to develop a new family of open source scoring systems for assessing the severity of illness of critical care patients internationally.


### What is APACHE?
The APACHE scoring systems have been developed to grade the severity of illness in critically ill patients. These systems are moderately accurate in predicting individual survival.

## Workflow
<img src="/img/Screen Shot 2020-02-07 at 1.51.07 PM.png"/>

## Why Feature Selection?
Although regularization helps constrain or shrinks our coefficient towards zero and help us to avoid overfitting;
Most of the times, we will have many non-informative features where poor-quality input will produce poor-quality output.

### Pearson Correlation
We check the absolute value of the Pearson’s correlation between the target and numerical features in our dataset. We keep the top n features based on this criteria. (categorical features should be encoding to 0/1)
```
'd1_calcium_min', 'h1_diasbp_min', 'h1_inr_min', 'd1_inr_min', 'h1_diasbp_noninvasive_min', 'd1_wbc_max', 'd1_mbp_invasive_min', 'h1_lactate_max', 'd1_albumin_max', 'ph_apache', 'h1_lactate_min', 'fio2_apache', 'd1_sysbp_invasive_min', 'h1_mbp_min', 'albumin_apache', 'd1_inr_max', 'h1_inr_max', 'h1_mbp_noninvasive_min', 'h1_sysbp_min', 'h1_sysbp_noninvasive_min', 'temp_apache', 'd1_albumin_min', 'd1_hco3_min', 'd1_heartrate_max', 'bun_apache', 'intubated_apache_0.0', 'intubated_apache_1.0', 'd1_bun_min', 'd1_arterial_ph_min', 'd1_bun_max', 'd1_diasbp_noninvasive_min', 'd1_diasbp_min', 'gcs_eyes_apache_4', 'd1_mbp_noninvasive_min', 'd1_mbp_min', 'd1_temp_min', 'gcs_verbal_apache_5', 'd1_spo2_min', 'd1_sysbp_noninvasive_min', 'd1_sysbp_min', 'gcs_verbal_apache_1', 'ventilated_apache_1.0', 'ventilated_apache_0.0', 'gcs_motor_apache_6', 'gcs_motor_apache_1', 'gcs_eyes_apache_1', 'd1_lactate_max', 'apache_4a_icu_death_prob', 'd1_lactate_min', 'apache_4a_hospital_death_prob'
```
<img src="/img/corr.png"/>

*For Example:*
- *bun_apache: It's a common blood test, the blood urea nitrogen (BUN) test reveals important information about how well your   kidneys and liver are working.*
- *intubated_apache: Whether the patient was intubated at the time of the highest scoring arterial blood test used in the oxygenation score*

### Chi-Squared
Higher the Chi-Square value the feature is more dependent on the response and it can be selected for model training. Data should be normalize.
sklearn.feature_selection.SelectKBest with chi2 as parameter.
```
['elective_surgery', 'apache_2_diagnosis', 'apache_post_operative', 'bun_apache', 'd1_diasbp_min', 'd1_diasbp_noninvasive_min', 'd1_heartrate_max', 'd1_mbp_min', 'd1_mbp_noninvasive_min', 'd1_sysbp_min', 'd1_sysbp_noninvasive_min', 'h1_sysbp_min', 'd1_bun_max', 'd1_bun_min', 'd1_creatinine_max', 'd1_lactate_max', 'd1_lactate_min', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'gcs_eyes_apache_0', 'gcs_eyes_apache_1', 'gcs_eyes_apache_2', 'gcs_eyes_apache_4', 'gcs_motor_apache_0', 'gcs_motor_apache_1', 'gcs_motor_apache_2', 'gcs_motor_apache_3', 'gcs_motor_apache_4', 'gcs_motor_apache_5', 'gcs_motor_apache_6', 'gcs_verbal_apache_0', 'gcs_verbal_apache_1', 'gcs_verbal_apache_5', 'gcs_unable_apache_1.0', 'intubated_apache_0.0', 'intubated_apache_1.0', 'ventilated_apache_0.0', 'ventilated_apache_1.0', 'immunosuppression_1.0', 'solid_tumor_with_metastasis_1.0', 'hospital_admit_source_Floor', 'hospital_admit_source_Operating Room', 'hospital_admit_source_Step-Down Unit (SDU)', 'icu_admit_source_Floor', 'icu_admit_source_Operating Room / Recovery', 'apache_3j_bodysystem_Metabolic', 'apache_3j_bodysystem_Sepsis', 'apache_2_bodysystem_Cardiovascular', 'apache_2_bodysystem_Metabolic', 'apache_2_bodysystem_Undefined diagnoses']
```
<img src="/img/select k_best_f_classif.png"/>

### Recursive Feature Elimination (RFE)
*From sklearn Documentation:*

*The goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.*

### Tree-based
We can also use RandomForest to select features based on feature importance.
We calculate feature importance using node impurities in each decision tree. In Random forest, the final feature importance is the average of all decision tree feature importance.

### Chossing our best features from our feature selection techniques
<img src="/img/o.png"/>

##  Machine Learning Models 
<img src="/img/roc.png"/>
<img src="/img/confussion_matrix_first_features.png"/>

### Analysis on Results

<img src="/img/apache_3j_bodysystem_plot.png"/>

*apache_3j_bodysystem: Admission diagnosis group for APACHE III*

## Next steps and Lessons learnt
Try Grid Search with one hyperparamter at a time.
Keep going with feature engineering and Gradient Boosting to make a better prediction.
