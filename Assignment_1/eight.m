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
            else
                data_n(i,6) = 1;
            end
        end
    end
    if n == 2
        for i = 1:150
            if data_n(i,6) ~= 2
                data_n(i,6) = 0;
            else
                data_n(i,6) = 1;
            end
        end
    end
    if n == 3
        for i = 1:150
            if data_n(i,6) ~= 3
                data_n(i,6) = 0;
            else
                data_n(i,6) = 1;
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
    y_p1 = 1/(1 + exp(-(x_test(i,:)*w_m(1, :)')));
    y_p2 = 1/(1 + exp(-(x_test(i,:)*w_m(2, :)')));
    y_p3 = 1/(1 + exp(-(x_test(i,:)*w_m(3, :)')));
    [val, y_pred(i)] = max([y_p1 y_p2 y_p3]);
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
    y_p1 = 1/(1 + exp(-(x_test(i,:)*w_m(1, :)')));
    y_p2 = 1/(1 + exp(-(x_test(i,:)*w_m(2, :)')));
    y_p3 = 1/(1 + exp(-(x_test(i,:)*w_m(3, :)')));
    if y_p1 == 0
        y_p1 = 1;
    else
        y_p1 = 2;
    end
    if y_p2 == 0
        y_p2 = 1;
    else
        y_p2 = 3;
    end
    if y_p3 == 0
        y_p3 = 2;
    else
        y_p3 = 3;
    end
    y_pred(i) = mode([y_p1 y_p2 y_p3]);
end
y_onevone = y_pred;
% Multiclass accuracy measures
U_onevall = zeros(3);
U_onevone = zeros(3);
for i = 1:60
    % Fill One v All
    for a = 1:3
        for b = 1:3
            if y_onevall(i) == a && y_test(i) == b
                U_onevall(a,b) = U_onevall(a,b) + 1;
            end    
        end
    end
    % Fill One v One
    for a = 1:3
        for b = 1:3
            if y_onevone(i) == a && y_test(i) == b
                U_onevone(a,b) = U_onevone(a,b) + 1;
            end    
        end
    end
end
oa_onevall = trace(U_onevall)/sum(sum(U_onevall));
ia_onevall = diag(U_onevall)/sum(sum(U_onevall));
disp('One vs All');
disp('Individual Accuracies');
disp(ia_onevall);
disp('Overall Accuracy');
disp(oa_onevall);

oa_onevone = trace(U_onevone)/sum(sum(U_onevone));
ia_onevone = diag(U_onevone)/sum(sum(U_onevone));
disp('One vs One');
disp('Individual Accuracies');
disp(ia_onevone);
disp('Overall Accuracy');
disp(oa_onevone);