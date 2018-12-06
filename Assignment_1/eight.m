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
% --- One vs All --- %
% Initialize weights matrix
w_m = zeros(3, 5);
for n = 1:3
    data_n = data;
    if n == 1
        for i = 1:150
            if data_n(i,6) ~= 1
                data_n(i,6) = 0;
            end
        end
    end
    if n == 1
        for i = 1:150
            if data_n(i,6) ~= 2
                data_n(i,6) = 0;
            end
        end
    end
    if n == 3
        for i = 1:150
            if data_n(i,6) ~= 3
                data_n(i,6) = 0;
            end
        end
    end
    shuffled = data_n(randperm(size(data_n,1)),:);
    x = shuffled(:,1:5);
    y = shuffled(:,6);
    x_train = x(1:90,:);
    y_train = y(1:90,:);
% Perform regression
    w_m(n, :) = logistic_regression(x_train, y_train, 100, 0.1);
end
% Make Test set
shuffled = data(randperm(size(data,1)),:);
x = shuffled(:,1:5);
y = shuffled(:,6);
x_test = x(91:150,:);
y_test = y(91:150,:);
% Validate
y_pred = zeros(size(y_test));
for i = 1:size(y_pred,1)
    y_p1 = round(1 + 1/(1 - exp(-(x_test(i,:)*w_m(1, :)'))));
    y_p2 = round(1 + 1/(1 - exp(-(x_test(i,:)*w_m(2, :)'))));
    y_p3 = round(1 + 1/(1 - exp(-(x_test(i,:)*w_m(3, :)'))));
    y_pred(i) = max([y_p1 y_p2 y_p3]);
end
y_onevall = y_pred;

% --- One vs One --- %
% Initialize weights matrix
w_m = zeros(3, 5);
for n = 1:3
    if n == 1
        data_n = data(1:100, :);
    elseif n == 2
        data_n = data([1:50 101:150], :);
    elseif n == 3
        data_n = data(51:150, :);
    end
    shuffled = data_n(randperm(size(data_n,1)),:);
    x = shuffled(:,1:5);
    y = shuffled(:,6);
    x_train = x(1:60,:);
    y_train = y(1:60,:);
% Perform regression
    w_m(n, :) = logistic_regression(x_train, y_train, 100, 0.1);
end
% Make Test set
shuffled = data(randperm(size(data,1)),:);
x = shuffled(:,1:5);
y = shuffled(:,6);
x_test = x(91:150,:);
y_test = y(91:150,:);
% Validate
y_pred = zeros(size(y_test));
for i = 1:size(y_pred,1)
    y_p1 = round(1 + 1/(1 - exp(-(x_test(i,:)*w_m(1, :)'))));
    y_p2 = round(1 + 1/(1 - exp(-(x_test(i,:)*w_m(2, :)'))));
    y_p3 = round(1 + 1/(1 - exp(-(x_test(i,:)*w_m(3, :)'))));
    y_pred(i) = mode([y_p1 y_p2 y_p3]);
end
y_onevone = y_pred;