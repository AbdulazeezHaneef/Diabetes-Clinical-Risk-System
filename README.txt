## Overview
This project aims to highlight the advantage of non linear models [Random Forest] over logistics regression. 

## Dataset
This dataset comes from anonymized clinical files and it includes a set of patient informations and lab work such as Creatinine and BUN levels and a breakdown of their existing
medical conditions

## Models Used
- Logistic Regression
- Random Forest Classifier

## Files
- diabetes_model.py : Main training and evaluation script
- diabetes_data.csv : Clinical dataset
- Results summary
- README
- diabetes_logistic_model.pkl : Trained logistic regression model
- diabetes_random_forest_model.pkl : Trained random forest model
- diabetes_scaler.pkl : Feature scaler

## Key Findings 
When applied, the Random Forest classifier showed glucosse levels as the main predictor of diabetes. This was then followed by Body Mass Index [BMI] and age. This perfectly corrolate with
the accepted and established clinical understanding of diabetis therapy. which shows that obesity-driven insulin resistance, chronic hyperglycemia are primary risk factors in additon to the
fact that over time the body's ability to process glucose declines.
The Random Forest Model used achieved a higer accuracy of prediction [76%] as compared to the logistics regression [71%] with better recall for diabetes cases. This shows that non-Linear models
have an advantage over logistics regression in analysing complex interactions [Metabolic].

## Author
Abdulazeez Hanif

## Note
Trained models are generated locally and are not stored in the repository.
 