clc;
clear;

% Standardizing function
normalize = @(v) (v-mean(v))/std(v);
% Sigmoid activation
sigmoid = @(x) 1./(1 + exp(-x));
%--------------------------------------%

% No. of hidden neurons
H = 20;

% Learning rate
lr = 0.25;

% Momentum factor
p = 0.025;

% Number of iterations
iterations = 1000;

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

% Hold-out data splitting
x_train = X(1:105, :);
x_test = X(106:150, :);
y_train = Y(1:105, :);
y_test = Y(106:150, :);
[M, N] = size(x_train);
[P, Q] = size(x_test);

% Number of classes
K = 3;

% --- MFNN --- %
% Initialize weights and bias(random value between -0.01 to 0.01)
w1 = (rand([N+1 H]) - rand([N+1 H]))/100;
w2 = (rand([H+1 3]) - rand([H+1 3]))/100;
b = -1;

% Auxiliaries
x_train = [b*ones(M, 1) x_train];
Dw1 = zeros(N+1, H);
Dw2 = zeros(H+1, 3);
cost = zeros([iterations 1]);

for k = 1:iterations
  % --- Forward propogation --- %
  z = [b*ones(M, 1) sigmoid(x_train*w1)];
  y = sigmoid(z*w2);
  % --- Back propogation --- %
  cost(k) = mean(mean((y_train - y).^2));
  df = y.*(1-y);
  d2 = df.*(y_train - y);
  Dw2 = (lr/N)*d2'*z;
  w2 = (1+p)*w2 + Dw2';
  
  df = z.*(1-z);
  d1 = df.*(d2*w2');
  d1 = d1(:, 2:end);
  Dw1 = (lr/N)*d1'*x_train;
  w1 = (1+p)*w1 + Dw1';
end

% --- RBFN --- %
% Number of RBF neurons
N = 30;
% Perform K-Means clustering on training data (for 10 neurons)
[l, mu] = kmeans(y, N);
% Apply RBF to get Hidden layer
for i = 1:size(y, 1)
    for j = size(mu, 1)
        % Cubic function
        H(i, j) = (norm(y(i, :) - mu(j, :)))^3;
    end
end
% Calculate weights
w = pinv(H)*y_train;

% --- Validation --- %
x_test = [b*ones(P, 1) x_test];
z_p1 = [b*ones(P, 1) sigmoid(x_test*w1)];
z_p2 = sigmoid(z_p1*w2);
for i = 1:size(z_p2, 1)
    for j = 1:size(mu, 1)
        H_t(i, j) = (norm(z_p2(i, :) - mu(j, :)))^3;
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
plot(cost);
disp(['Test accuracy: ', num2str(val_acc)]);