clc;
clear;

% Standardizing function
normalize = @(v) (v-mean(v))/std(v);
% Sigmoid activation
sigmoid = @(x) 1./(1 + exp(-x));
%--------------------------------------%

% No. of hidden neurons
H1 = 15;
H2 = 5;

% Learning rate
lr = 0.25;

% Momentum factor
p = 0.001;

% Number of iterations
iterations = 1500;

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

% Initialize weights and bias(random value between -0.01 to 0.01)
w1 = (rand([N+1 H1]) - rand([N+1 H1]))/100;
w2 = (rand([H1+1 H2]) - rand([H1+1 H2]))/100;
w3 = (rand([H2+1 3]) - rand([H2+1 3]))/100;
b = -1;

% Auxiliaries
x_train = [b*ones(M, 1) x_train];
Dw1 = zeros(N+1, H1);
Dw2 = zeros(H1+1, H2);
Dw3 = zeros(H2+1, 3);
cost = zeros([iterations 1]);

for k = 1:iterations
  % --- Forward propogation --- %
  z1 = [b*ones(M, 1) sigmoid(x_train*w1)];
  z2 = [b*ones(M, 1) sigmoid(z1*w2)];
  y = sigmoid(z2*w3);
  % --- Back propogation --- %
  cost(k) = mean(mean((y_train - y).^2));
  
  df = y.*(1-y);
  d3 = df.*(y_train - y);
  Dw3 = (lr/N)*d3'*z2;
  w3 = (1+p)*w3 + Dw3';
  
  df = z2.*(1-z2);
  d2 = df.*(d3*w3');
  d2 = d2(:, 2:end);
  Dw2 = (lr/N)*d2'*z1;
  w2 = (1+p)*w2+Dw2';
  
  df = z1.*(1-z1);
  d1 = df.*(d2*w2');
  d1 = d1(:, 2:end);
  Dw1 = (lr/N)*d1'*x_train;
  w1 = (1+p)*w1 + Dw1';
end
% --- Validation --- %
[m, n] = size(x_test);
x_test = [ones(m, 1) x_test];
z1_p = [b*ones(m, 1) sigmoid(x_test*w1)];
z2_p = [b*ones(m, 1) sigmoid(z1_p*w2)];
y_p = sigmoid(z2_p*w3);
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
for i = 1:45
    if y_p(i, :) == y_test(i, :)
        correct = correct + 1;
    end
end
val_acc = correct/45;
plot(cost);
disp(['Test accuracy: ', num2str(val_acc)]);