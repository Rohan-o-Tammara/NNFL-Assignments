clc;
clear;

% Standardizing function
normalize = @(v) (v-mean(v))/std(v);
%--------------------------------------%

% Load data
dat = csvread('dataset.csv');

% Normalize input data
X = [normalize(dat(:, 1)) normalize(dat(:, 2)) normalize(dat(:, 3)) normalize(dat(:, 4)) normalize(dat(:, 5)) normalize(dat(:, 6)) normalize(dat(:, 7))];

% One hot encode output classes
Y = ind2vec(dat(:, 8)')';

% Shuffle data
data = [X full(Y)];
data = data(randperm(150), :);
X = data(:, 1:7);
Y = data(:, 8:10);

% Hold-out data splitting (70:30)
x_train = X(1:105, :);
x_test = X(106:150, :);
y_train = Y(1:105, :);
y_test = Y(106:150, :);

% --- RBFN --- %
% Number of RBF neurons
N = 5;

% Perform K-Means clustering on training data (for 10 neurons)
[l, mu] = kmeans(x_train, N);
% Apply RBF to get Hidden layer
for i = 1:size(x_train, 1)
    for j = size(mu, 1)
        % Cubic function
        H(i, j) = (norm(x_train(i, :) - mu(j, :)))^3;
    end
end
% Calculate weights
w = pinv(H)*y_train;

% --- Validation --- %
for i = 1:size(x_test, 1)
    for j = 1:size(mu, 1)
        H_t(i, j) = (norm(x_test(i, :) - mu(j, :)))^3;
    end
end
y_p = H_t*w;
for i = 1:size(y_p, 1)
    [val, idx] = max(y_p(i, :));
    y_p(i, idx) = 1;
    for j = 1:3
        if y_p(i, j) ~= 1
            y_p(i, j) = 0;
        end
    end
end
correct = 0;
for i = 1:size(y_test, 1)
    if y_p(i, :) == y_test(i, :)
        correct = correct + 1;
    end
end
val_acc = correct/45;
disp(['Test accuracy: ', num2str(val_acc)]);