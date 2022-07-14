# Model Card

## Model Details
- Developed by Timo Mutas for the Udacity ML DevOps Nanodegree
- scikit-learn Random Forest Classifier 

## Intended Use
- Baseline classification model to exemplify an ML pipeline with serving the model as subsequent inference REST API endpoint
- Therefore the actual model and its performance are irrelevant for the project
- Do not use

## Training Data
- Using the Census data set from the UCI Machine Learning repository
- Task is to predict the income class based on various demographic data
- URL: https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
- Using sklearn's train_test_split, set 0.3 of the raw dataset aside for model evaluation

## Metrics
- No through analysis of model performance has been conducted, since the focus of this exercise was on MLOps

## Ethical Considerations
- The dataset contains sensitive personal information such as gender, race, marital status, etc.
- As not much is known about the acquisition of this data, there is uncertainty about any biases in the data, that can lead to discriminatory biases in predictions too.

## Caveats and Recommendations
- Do not take this model seriously in any way, as the main purpose of this repo was gaining experience on the pipeline and API part of MLOps.

Based on the model card template of Mitchell, et al., 2019 https://arxiv.org/pdf/1810.03993.pdf