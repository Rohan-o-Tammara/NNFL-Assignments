clc;
clear;

% Define normalizing function
normalize = @(v) (v-mean(v))/std(v);
% ---------------------------------- %

% Num of iterations
k = 500;
% Load data
data = csvread('data.csv', 1, 0);
x1 = data(:, 1);
x2 = data(:, 2);
y = data(:, 3);
% Initialize variables
x = [ones(349,1) normalize(x1) normalize(x2)];
[m, n] = size(x);
w = [randn(1, n); zeros(k-1, n)];
h = zeros(m, 1);
b = 0;
% Initialize cost vector
cost = zeros(k, 1);
% Learning rate (alpha)
lr = 0.01;
% Regularization parameter
lambda = 0.09;
% Start learning
for iter = 1:k
    % Update hypothesis
    h = x*w(iter,:)' + b;
    % Update cost
    cost(iter) =(1/2*m)*((h-y)'*(h-y)+lambda*w(iter,:)*w(iter,:)');
    % Update weights (Stochastic GD)
    for i = 1:m
        for j = 1:n
            w(iter,j) = (1-lr*lambda*(1/m))*w(iter,j) - lr*(1/m)*(h(i)-y(i))*x(i,j);
        end
    end
    w(iter+1,:) = w(iter,:);
end
% J vs k
figure(1)
plot(cost)