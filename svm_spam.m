%% Machine Learning Online Class
%% Support Vector Machines
%% This file is a development from the CS229 Standford class 
%% The SVM function from LIBSVM is used to gain speed and flexibility
%% LIBSVM needs to be installed on the computer before using the program
%% Dataset for applying SVM are those of CS229, but the tool works for any other set of data
%% Spam Classification with SVMs

%% Initialization
clear ; close all; clc
options_svm = ('-t 0, -g 0.1 ');       % define options (see below with first parameter = kernel type and secong is g or C
%% ==================== Part 1: Email Preprocessing ====================

fprintf('\nPreprocessing sample email (emailSample1.txt)\n');

% Extract Features
file_contents = readFile('emailSample1.txt');
word_indices  = svm_processEmail(file_contents);

% Print Stats
fprintf('Word Indices: \n');
fprintf(' %d', word_indices);
fprintf('\n\n');

fprintf('Program paused end Part 1. Press enter to continue.\n');
pause;

%% ==================== Part 2: Feature Extraction ====================

fprintf('\nExtracting features from sample email (emailSample1.txt)\n');

% Extract Features
file_contents = readFile('emailSample1.txt');
word_indices  = svm_processEmail(file_contents);
features      = svm_emailFeatures(word_indices);

% Print Stats
fprintf('Length of feature vector: %d\n', length(features));
fprintf('Number of non-zero entries: %d\n', sum(features > 0));

fprintf('Program paused end Part 2. Press enter to continue.\n');
pause;

%% =========== Part 3: Train Linear SVM for Spam Classification ========
load('spamTrain.mat');

fprintf('\nTraining Linear SVM (Spam Classification)\n')
fprintf('(this may take 1 to 2 minutes) ...\n')

C = 0.1;
model = svmtrain(y, X, options_svm);

p = svmpredict(y, X, model);

fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

%% =================== Part 4: Test Spam Classification ================
%  After training the classifier, we can evaluate it on a test set. We have
%  included a test set in spamTest.mat

% Load the test dataset
% You will have Xtest, ytest in your environment
load('spamTest.mat');

fprintf('\nEvaluating the trained Linear SVM on a test set ...\n')

p = svmpredict(y, Xtest, model);

fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);
pause;


%% ================= Part 5: Top Predictors of Spam ====================
%  Since the model we are training is a linear SVM, we can inspect the
%  weights learned by the model to understand better how it is determining
%  whether an email is spam or not. The following code finds the words with
%  the highest weights in the classifier. Informally, the classifier
%  'thinks' that these words are the most likely indicators of spam.
%

% Sort the weights and obtin the vocabulary list
[weight, idx] = sort(model.w, 'descend');
vocabList = getVocabList();

fprintf('\nTop predictors of spam: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end

fprintf('\n\n');
fprintf('\nProgram paused end part 5. Press enter to continue.\n');
pause;

%% =================== Part 6: Try Your Own Emails =====================
%  Now that you've trained the spam classifier, you can use it on your own
%  emails! In the starter code, we have included spamSample1.txt,
%  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
%  The following code reads in one of these emails and then uses your 
%  learned SVM classifier to determine whether the email is Spam or 
%  Not Spam

% Set the file to be read in (change this to spamSample2.txt,
% emailSample1.txt or emailSample2.txt to see different predictions on
% different emails types). Try your own emails as well!
filename = 'spamSample1.txt';

% Read and predict
file_contents = readFile(filename);
word_indices  = svm_processEmail(file_contents);
x             = svm_emailFeatures(word_indices);
p = svmpedict(y, x, model);

fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');


%% ==================== Options of SVM algorithm ====================
%%  libsvm_options:
%% -s svm_type : set type of SVM (default 0)
%%        0 -- C-SVC              (multi-class classification)
%%        1 -- nu-SVC             (multi-class classification)
%%        2 -- one-class SVM
%%        3 -- epsilon-SVR        (regression)
%%        4 -- nu-SVR             (regression)
%% -t kernel_type : set type of kernel function (default 2)
%%        0 -- linear: u'*v
%%        1 -- polynomial: (gamma*u'*v + coef0)^degree
%%        2 -- radial basis function: exp(-gamma*|u-v|^2) = GaussianKernel
%%        3 -- sigmoid: tanh(gamma*u'*v + coef0)
%%        4 -- precomputed kernel (kernel values in training_instance_matrix)
%% -d degree : set degree in kernel function (default 3)
%% -g gamma : set gamma in kernel function (default 1/num_features) = C ~1/lambda
%% -r coef0 : set coef0 in kernel function (default 0)
%% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
%% -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
%% -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
%% -m cachesize : set cache memory size in MB (default 100)
%% -e epsilon : set tolerance of termination criterion (default 0.001)
%% -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
%% -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
%% -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
%% -v n: n-fold cross validation mode
%% -q : quiet mode (no outputs)



