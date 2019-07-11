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
%

choice_list = [0.01,0.03,0.1,0.3,1,3,10,30];
optimal_loss = -1;
for i=1:length(choice_list)
    for j=1:length(choice_list)
        c = choice_list(i);
        sig = choice_list(j);
        % 借鉴ex6的part 7中svmTrain的调用方式：
        % model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        % 现在C(c)和sigma(sig)以及选好，X和y已知，可以进行训练，并得到模型model了
        model = svmTrain(X,y,c,@(x1,x2) gaussianKernel(x1,x2,sig));
        % 用训练好的模型在验证集上预测一次
        predictions = svmPredict(model,Xval);
        % 根据预测结果predictions和真实标签yval计算验证集上的损失
        loss = mean(double(predictions~=yval));
        % 如果此时的损失更小，说明本次的C和sigma组合更好，则记录下来
        if optimal_loss==-1 || loss<optimal_loss
            optimal_loss = loss;
            C = c;
            sigma = sig;
        end
    end
end


% =========================================================================

end
