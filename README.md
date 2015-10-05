[![Build Status](https://travis-ci.org/alex-pirozhenko/sklearn-pmml.svg)](https://travis-ci.org/alex-pirozhenko/sklearn-pmml)
[![Gitter](https://badges.gitter.im/alex-pirozhenko/sklearn-pmml.svg)](https://gitter.im/alex-pirozhenko/sklearn-pmml)

# sklearn-pmml
A library that allows serialization of SciKit-Learn estimators into PMML

# Installation
The easiest way is to use pip:
```
pip install sklearn-pmml
```

# Supported models
- DecisionTreeClassifier
- DecisionTreeRegressor
- GradientBoostingClassifier
- RandomForestClassifier

# PMML output

## Classification
Classifier converters can only operate with categorical outputs, and for each categorical output variable ```varname``` 
the PMML output contains the following outputs:
- categorical ```varname``` for the predicted label of the instance
- double ```varname::label``` for the probability for a given label

## Regression
Regression model PMML outputs the numeric response variable named as the output variable
