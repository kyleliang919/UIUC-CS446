#!/bin/bash

mkdir bin

make

# Generate the example features (first and last characters of the
# first names) from the entire dataset. This shows an example of how
# the feature files may be built. Note that don't necessarily have to
# use Java for this step.

#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.all ./../badges.example.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold1 ./../badges.fold1.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold2 ./../badges.fold2.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold3 ./../badges.fold3.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold4 ./../badges.fold4.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold5 ./../badges.fold5.arff
# Using the features generated above, train a decision tree classifier
# to predict the data. This is just an example code and in the
# homework, you should perform five fold cross-validation. 
java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.fold1.arff ./../badges.fold2.arff ./../badges.fold3.arff ./../badges.fold4.arff ./../badges.fold5.arff
