# About
This is a simple [JPMML](http://github.com/jpmml)-based CLI evaluator for PMML models.

# Notes
This submodule relies on AGPL library [jpmml-evaluator](http://github.com/jpmml/jpmml-evaluator), 
but it's only used for testing and it's not a part of sklearn-pmml distribution.
Since users will not interact with AGPL-licensed library, I think it's OK to use it in tests.
 
# Usage
1. Build the JAR file (make sure you have JDK8 installed):
```
mvn clean package
```
2. Run with maven:
```
mvn exec:java -e -q \
-Dexec.mainClass=sklearn.pmml.jpmml.JPMMLCSVEvaluator \
-Dexec.args=/path/to/pmml /path/to/input.csv /path/to/output.csv 
```