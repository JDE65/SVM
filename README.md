# SVM
Support Vector Machine using LIBSVM library
LIBSVM has to be first installed on the computer to use the code. - https://www.csie.ntu.edu.tw/~cjlin/libsvm/oldfiles/index-1.0.html
SVM is a learning algorithm that analyze data used for classification and regression analysis.
The library contains 3 master codes that uses sub-functions, all available here :
- svm.m : use SVM algorithm to categorize data from a template dataset
- svmcv : use SVM algorithm with a dfined training set size and a cross-validation set. It computes the accuracy, the precision, recall and F1 of both training and cross-val data
- svmcv_trainsize : does the same as svmcv but run for various size of training set to illustrate the progress in accuracy and F1
- svm_spam : use SVM to check if a mail is a spam, based on a library. Both the template email and the library can be modified

