function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))

val = [.01 .03 .1 .3 1 3 10 30];
err = 0;
err_min = 999999999999;
for C = val;
  for sigma = val;
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    %err = err + (svmPredict(model, Xval) - yval);
    err = mean(double(svmPredict(model, Xval) != yval));
    if (err <= err_min)
      C_fin = C;
      s_fin = sigma;
      err_min = err;
    endif
  endfor
endfor

C = C_fin;
sigma = s_fin;



% =========================================================================

end
