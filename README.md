[![Build Status](https://travis-ci.org/alex-pirozhenko/sklearn-pmml.svg)](https://travis-ci.org/alex-pirozhenko/sklearn-pmml)

# sklearn-pmml
A library that allows serialization of SciKit-Learn estimators into PMML

# Supported models
- DecisionTreeClassifier
- DecisionTreeRegressor
- GradientBoostingClassifier
- RandomForestClassifier

# PMML output

## Classification
Classifier converters can only operate with categorical outputs, and for each categorical output variable ```varname``` 
the PMML output contains the following outputs:

| Output | Type | Description |
-------------------------------
| varname | categorical | label for the instance |
| varname::label | double | probability for a given label |

## Regression
Regression model PMML outputs the numeric response variable named as the output variable
