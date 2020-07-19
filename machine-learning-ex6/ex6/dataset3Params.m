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
r = [0.010000    0.030000    0.100000    0.300000    1.000000    3.000000   10.000000   30.000000];
s = zeros(64,2);
for i = 1:8
    for j = 1:8
        s(((i-1)*8) + j ,:) = r([i j]);
    endfor
endfor
errs = Inf(64,1);
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
%

for i = 1:size(s,1)
    p = s(i,:);
    model= svmTrain(X, y, p(1), @(x1, x2) gaussianKernel(x1, x2, p(2)));
    predictions = svmPredict(model, Xval);
    errs(i) = mean(double(predictions ~= yval));
endfor

[v , j] = min(errs);
C = s(j, 1);
sigma = s(j, 2);
% =========================================================================

end
