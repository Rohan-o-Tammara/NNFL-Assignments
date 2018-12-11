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
    % --- One vs All --- %
    % Initialize weights matrix
    w_m = zeros(3, 5);
    % Prepare training labels
    for n = 1:3
        y_n = y_train;
        if n == 1
            for i = 1:120
                if y_n(i) ~= 1
                    y_n(i) = 0;
                else
                    y_n(i) = 1;
                end
            end
        end
        if n == 2
            for i = 1:120
                if y_n(i) ~= 2
                    y_n(i) = 0;
                else
                    y_n(i) = 1;
                end
            end
        end
        if n == 3
            for i = 1:120
                if y_n(i) ~= 3
                    y_n(i) = 0;
                else
                    y_n(i) = 1;
                end
            end
        end
        % Prepare test labels
        y_nt = y_test;
        if n == 1
            for i = 1:30
                if y_nt(i) ~= 1
                    y_nt(i) = 0;
                else
                    y_nt(i) = 1;
                end
            end
        end
        if n == 2
            for i = 1:30
                if y_nt(i) ~= 2
                    y_nt(i) = 0;
                else
                    y_nt(i) = 1;
                end
            end
        end
        if n == 3
            for i = 1:30
                if y_nt(i) ~= 3
                    y_nt(i) = 0;
                else
                    y_nt(i) = 1;
                end
            end
        end
    % Perform regression
        w_m(n, :) = logistic_regression(x_train, y_n, 100, 0.1);
    end
    % Validate
    y_pred = zeros(size(y_nt));
    for i = 1:size(y_pred,1)
        y_p1 = 1/(1 + exp(-(x_test(i,:)*w_m(1, :)')));
        y_p2 = 1/(1 + exp(-(x_test(i,:)*w_m(2, :)')));
        y_p3 = 1/(1 + exp(-(x_test(i,:)*w_m(3, :)')));
        [val, y_pred(i)] = max([y_p1 y_p2 y_p3]);
    end
    % Multiclass accuracy measures
    U = zeros(3);
    for i = 1:30
        % Fill
        for a = 1:3
            for b = 1:3
                if y_pred(i) == a && y_nt(i) == b
                    U(a,b) = U(a,b) + 1;
                end    
            end
        end
    end
    oa = trace(U)/sum(sum(U));
    ia = diag(U)/sum(sum(U));
    disp(['Fold ', num2str(f)]);
    disp('Individual Accuracies');
    disp(ia);
    disp('Overall Accuracy');
    disp(oa);
end