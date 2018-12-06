clc;
clear;

% Standardizing function
normalize = @(v) (v-mean(v))/std(v);
% Sigmoid activation
sigmoid = @(x) 1/(1 + exp(-x));
%--------------------------------------%

% No. of hidden neurons
H = 32;

% Learning rate
lr = 0.5;

% Number of iterations
iterations = 1000;

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

% Hold-out data splitting
x_train = X(1:105, :);
x_test = X(106:150, :);
y_train = Y(1:105, :);
y_test = Y(106:150, :);
[M, N] = size(x_train);
[P, Q] = size(x_test);

% Number of classes
K = 3;

% Initialize weights (random value between -0.01 to 0.01)
w1 = (rand([H 7]) - rand([H 7]))/100;
w2 = (rand([3 H]) - rand([3 H]))/100;
b1 = (rand - rand)/100;
b2 = (rand - rand)/100;

% Initialize intermediates (for our ease)
z = zeros([1 H]);
y = zeros([1 K]);
del_w1 = zeros([H 7]);
del_w2 = zeros([3 H]);

cost = zeros([iterations 1]);

z_test = zeros([1 H]);
y_pred = zeros([P K]);

% Start training
for k = 1:iterations
    for t = 1:M
        % -- Forward Propogation -- %
        for h = 1:H
            z(h) = sigmoid(sum(w1(h, :).*x_train(t, :)) + b1);
        end
        
        for i = 1:K
            y(t, i) = sigmoid(sum(w2(i, :).*z) + b2);
        end
        % -- Back Propogation -- %
        for i = 1:K
            cost(k) = cost(k) + (y_train(t, i) - y(t, i))^2;
        end
        cost(k) =  0.5*sqrt(cost(k)/K);
        for i = 1:K
            for h = 1:H
                del_w2(i, h) = -lr*(y_train(t, i)-y(t, i))*y(t, i)*(1-y(t,i))*z(h);
            end
            del_b2 = -lr*(y_train(t, i)-y(t, i))*y(t, i)*(1-y(t,i));
        end
        
        for h = 1:H
            for j = 1:N
                sigma = 0;
                for i = 1:K
                    sigma = sigma + (y_train(t,i)-y(t,i))*w2(i,h);
                end
                del_w1(h, j) = -lr*sigma*z(h)*(1-z(h))*x_train(t,j);
            end
            del_b1 = -lr*sigma*z(h)*(1-z(h));
        end
        
        for i = 1:K
            for h = 1:H
                w2(i,h) = w2(i,h) - del_w2(i,h);
            end
            b2 = b2 - del_b2;
        end
        
        for h = 1:H
            for j = 1:N
                w1(h,j) = w1(h,j) - del_w1(h, j);
            end
            b1 = b1 - del_b1;
        end
    end  
    % --- Validation --- %
    for p = 1:P
        for h = 1:H
            z_test(h) = sigmoid(sum(w1(h, :).*x_test(p, :)) + b1);
        end
        for i = 1:K
            y_pred(p, i) = sigmoid(sum(w2(i, :).*z_test) + b2);
            if y_pred(p, i) > 0.5
                y_pred(p, i) = 1;
            else
                y_pred(p, i) = 0;
            end
        end
    end
    correct = 0;
    for i = 1:45
        if y_pred(i, :) == y_test(i, :)
            correct = correct + 1;
        end
    end
    val_acc = correct/45;
end
plot(cost);
disp(['Test accuracy: ', num2str(val_acc)]);