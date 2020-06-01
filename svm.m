%% Machine Learning Online Class
%% Support Vector Machines
%% This file is a development from the CS229 Standford class 
%% The SVM function from LIBSVM is used to gain speed and flexibility
%% LIBSVM needs to be installed on the computer before using the program
%% Dataset for applying SVM are those of CS229, but the tool works for any other set of data

%% Initialization
clear ; close all; clc

%% =============== Part 0: Input zone ================
data_file = 'data2.mat';           % choose the dataset for training and cross-validation
options_svm = ('-t 2, -g 10 ');       % define options (see below with first parameter = kernel type and secong is g or C
% svm_kernel_type = 2;                % For details, see options at the bottom
% C = 100;                              % 0 if default value to be chosen = 1 / nbr of features
test_kernel = 0;                      % if 1 => test 4 types of kernel, if not, using svm_kernel_type


%% =============== Part 1: Loading and Visualizing Data ================

load(data_file);
% plot_svm(X, y);
% if (C == 0);
%  C = 1 / size(X, 2);
% endif

%% =============== Part 2: Training and Visualizing Data ================

fprintf('\nTraining SVM Vizualize boundaries ...\n')
if (test_kernel == 1);
Effic = zeros(1, 4);
for i= 1:4;
  model = svmtrain(y, X, options_svm);
  visualizeBoundary_svm(X, y, model);
  [predict_label, accuracy, prob_estimates] = svmpredict(y, X, model);
  Effic = [Effic; i, accuracy'];
endfor
Effic
else
%  options_svm = ('-t svm_kernel_type, -g C ');
   model = svmtrain(y, X, options_svm);
   visualizeBoundary_svm(X, y, model);
  [predict_label, accuracy, prob_estimates] = svmpredict(y, X, model);
endif


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



