clc;
clear;

% Standardizing function
normalize = @(v) (v-mean(v))/std(v);
% ---------------------------- %

% Load data
raw = csvread('data4.csv', 1, 0);
x1 = raw(:, 1);
x2 = raw(:, 2);
x3 = raw(:, 3);
x4 = raw(:, 4);
y = raw(:, 5);
% Prepare data
data = [ones(150,1) normalize(x1) normalize(x2) normalize(x3) normalize(x4) y];
% Initialize weights
w = rand(1, 5);
% 5-fold cross validation
for f = 1:5
    shuffled = data(randperm(size(data,1)),:);
    x = shuffled(:,1:5);
    y = shuffled(:,6);
    if f == 1
        x_train = x(31:150, :);
        y_train = y(31:150, :);
        x_test = x(1:30, :);
        y_test = y(1:30, :);
    elseif f == 2
        x_train = x([1:30 61:150], :);
        y_train = y([1:30 61:150], :);
        x_test = x(31:60, :);
        y_test = y(1:30, :);        
    elseif f == 3
        x_train = x([1:60 91:150], :);
        y_train = y([1:60 91:150], :);
        x_test = x(61:90, :);
        y_test = y(61:90, :);
    elseif f == 4
        x_train = x([1:90 121:150], :);
        y_train = y([1:90 121:150], :);
        x_test = x(91:120, :);
        y_test = y(91:120, :);
    elseif f == 5
        x_train = x(1:120, :);
        y_train = y(1:120, :);
        x_test = x(121:150, :);
        y_test = y(121:150, :);
    end
    w = logistic_regression(x_train, y_train, 100, 0.1, w);
    % Validation
    y_pred = zeros(size(y_test));
    for i = 1:size(y_pred,1)
        y_pred(i) = round(1 + 1/(1 - exp(-(x_test(i,:)*w'))));
    end
end