clc;
clear;

% Normalizing function
normalize = @(v) (v-mean(v))/std(v);
% ----------------------------- %

% Load data
data = csvread('data2.csv', 1, 0);
x1 = data(:, 1);
x2 = data(:, 2);
x3 = data(:, 3);
x4 = data(:, 4);
% Data with added labels column
x = [normalize(x1) normalize(x2) normalize(x3) normalize(x4)];
[p, q] = size(x);
x = [x zeros(p,1)];
% Clusters initialized
n = 2;
c = rand(n, q);
% Number of iterations
k = 10;
% Start K-Means
for iter = 1:k
    % Update clusters
    for i = 1:p
        dist = zeros(n,1);
        for j = 1:n
            dist(j) = sqrt((x(i,1:q)-c(j,:))*(x(i,1:q)-c(j,:))');
        end
        x(i,q+1) = find(dist == min(dist));
    end
    % Update centers
    for j = 1:n
        c(j,:) = mean(x(x(:,q+1)==j,1:q));
    end
end
plot()
