%% Machine Learning Online Class
%% Support Vector Machines
%% This file is a development from the CS229 Standford class 
%% The SVM function from LIBSVM is used to gain speed and flexibility
%% LIBSVM needs to be installed on the computer before using the program
%% Dataset for applying SVM are those of CS229, but the tool works for any other set of data

%% Initialization
% clear ; close all; clc

%% =============== Part 0: Input zone ================
data_file = 'SVMv4A.mat';           % choose the dataset for training and cross-validation
train_size = 2000;
cv_size_init = 2001;
options_svm = ('-t 2, -g 20 ');       % define options (see below with first parameter = kernel type and secong is g or C
% svm_kernel_type = 2;                % For details, see options at the bottom
% C = 100;                              % 0 if default value to be chosen = 1 / nbr of features
test_kernel = 0;                      % if 1 => test 4 types of kernel, if not, using svm_kernel_type


%% =============== Part 1: Loading and Visualizing Data ================

load(data_file);
% plot_svm(X, y);
% if (C == 0);
%  C = 1 / size(X, 2);
% endif

Xnorm = nnormalizeX(X);
X = Xnorm;
sel = randperm(size(X, 1));
X = X(sel, :);
y = y(sel, 1);
m = size(X, 1);                     % size of X as complete matrix 
if m < train_size                   % constraining the size of the training set
   train_size = m;
endif
if m < cv_size_init                   % constraining the size of the cross-validation set
   cv_size_init = m - 100;
endif
%% Define X for training and for test 

X_train = X(1:train_size , :);      % define part of X used for training set
y_train = y(1:train_size , 1);      % define part of y used for training set
m_train = size(X_train, 1);         % define size of X used for raining set

X_test = X(cv_size_init:end , :);   % define part of X for cross_validation, starting from initial point
y_test = y(cv_size_init:end , 1);   % define part of y for cross_validation
m_test = size(X_test, 1);             % define size of X used for cross_validation


%% =============== Part 2: Training and Visualizing Data ================

fprintf('\nTraining SVM Vizualize boundaries ...\n')
% if (test_kernel == 1);
% Effic = zeros(1, 4);
% for i= 1:4;
%  model = svmtrain(y_train, X_train, options_svm);
%  visualizeBoundary_svm(X_train, y_train, model);
% [predict_label, accuracy, prob_estimates] = svmpredict(y_train, X_train, model);
%  Effic = [Effic; i, accuracy'];
% endfor
% Effic
% else
%  options_svm = ('-t svm_kernel_type, -g C ');
   model = svmtrain(y, X, options_svm);
  % visualizeBoundary_svm(X, y, model);
  [pred, accuracy, prob_estimates] = svmpredict(y_train, X_train, model); 

% endif

actP= sum(y_train(:)(y_train(:)==1));         % compute the number of negative of y
actN = m_train - actP;                  % compute the number of negative of y
predP = sum(pred(:)(pred(:)==1));
predN = m_train - predP;
TN=0;
for i=1:m_train;
  if y_train(i,1)==pred(i,1) && y_train(i,1)==0;
    TN = TN + 1;
  endif
endfor
FN = predN - TN;
FP = actN - TN;
TP = actP - FN;
Recall = TP / (TP + FN);
Precis = TP / (TP + FP);
F1 = 2 * Precis * Recall / (Precis + Recall);

%% =============== Part 3: Training the cross-validation Data ================

[pred_test, accuracy, prob_estimates] = svmpredict(y_test, X_test, model); 
actP= sum(y_test(:)(y_test(:)==1));         % compute the number of negative of y
actN = m_test - actP;                  % compute the number of negative of y
predP = sum(pred_test(:)(pred_test(:)==1));
predN = m_test - predP;
TNt=0;
for i=1:m_test;
  if y_test(i,1)==pred_test(i,1) && y_test(i,1)==0;
    TNt = TNt + 1;
  endif
endfor
FNt = predN - TNt;
FPt = actN - TNt;
TPt = actP - FNt;
Recallt = TPt / (TPt + FNt);
Precist = TPt / (TPt + FPt);
F1t = 2 * Precist * Recallt / (Precist + Recallt);

%% =============== Part 4: Printing results ================

fprintf('Size training                 : %f\n', m_train);
fprintf('Training Accuracy of the SVM  : %f\n', mean(double(pred == y_train)) * 100);
fprintf('Precision training            : %f\n', Precis);
fprintf('Recall training               : %f\n', Recall);
fprintf('F1 training                   : %f\n', F1t);
fprintf('=> True positives trained     : %f\n', TP);
fprintf('=> False positives trained    : %f\n', FP);
fprintf('=> False negatives trained    : %f\n', FN);


fprintf('\n  Size cross-validation         : %f\n', m_test);
fprintf('  Cross-validation Accuracy     : %f\n', mean(double(pred_test == y_test)) * 100);
fprintf('  Precision cross-val           : %f\n', Precist);
fprintf('  Recall cross-val              : %f\n', Recallt);
fprintf('  F1 cross-val                  : %f\n', F1t);
fprintf('  => True positives test        : %f\n', TPt);
fprintf('  => False positives test       : %f\n', FPt);
fprintf('  => False negatives test       : %f\n', FNt);


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



