clc;
clear;

% Standardizing function
normalize = @(v) (v-mean(v))/std(v);
%--------------------------------------%

% Load data
dat = csvread('dataset.csv');

% Normalize input data
X = [normalize(dat(:, 1)) normalize(dat(:, 2)) normalize(dat(:, 3)) normalize(dat(:, 4))...
    normalize(dat(:, 5)) normalize(dat(:, 6)) normalize(dat(:, 7))];

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
y_train = full(Y(1:105, :));
y_test = full(Y(106:150, :));

% --- RBFN --- %
% Number of RBF neurons
N = 15;

% Perform K-Means clustering on training data (for 10 neurons)
[~, mu] = kmeans(x_train, N);
% Apply RBF to get Hidden layer
for i = 1:size(x_train, 1)
    for j = 1:size(mu, 1)
        % Cubic function
        H(i, j) = (norm(x_train(i, :) - mu(j, :)))^3;
    end
end
% Calculate weights
w = pinv(H)*y_train;

% --- Validation --- %
for i = 1:size(x_test, 1)
    for j = 1:size(mu, 1)
        Ht(i, j) = (norm(x_test(i, :) - mu(j, :)))^3;
    end
end
y_p = Ht*w;
pred = zeros(1, 45);
orig = zeros(1, 45);

for i = 1:size(y_p, 1)
    [~, pred(i)] = max(y_p(i, :));
    [~, orig(i)] = max(y_test(i, :));
end
cm = confusionmat(orig, pred);
acc = trace(cm)/sum(sum(cm));
disp(['Accuracy: ', num2str(acc)]);