clc;
clear;

% Sigmoid activation
sigmoid = @(x) 1/(1 + exp(-x));
%--------------------------------------%

% Load data and shuffle it
load('data_for_cnn.mat');
load('class_label.mat');
data = [ecg_in_window label];
data = data(randperm(1000), :);
x = data(:, 1:1000);
y = data(:, 1001);

% No. of hidden neurons
H1 = 40;
H2 = 20;

% Learning rate
lr = 0.1;

% Number of iterations
iterations = 5;

% Initialize Kernel
g = [1/3 1/3 1/3];

% Initialize weights
w1 = (rand([H1 499]) - rand([H1 499]))/100;
w2 = (rand([H2 H1]) - rand([H2 H1]))/100;
w3 = (rand([1 H2]) - rand([1 H2]))/100;
b1 = (rand - rand)/100;
b2 = (rand - rand)/100;
b3 = (rand - rand)/100;

% Initialize intermediaries
z1 = zeros([1 H1]);
z2 = zeros([1 H2]);
y_p = zeros([1000 1]);
del_w1 = zeros([H1 1000]);
del_w2 = zeros([H2 H1]);
del_w3 = zeros([1 H2]);
cost = zeros([1 iterations]);

% Start Training
for k = 1:iterations
    for m = 1:1000
        cost(k) = cost(k) + (y_p(m) - y(m)).^2;
    end
    cost(k) = 0.5*sqrt(cost(k));
    % ---- Forward Propogation --- %
    for m = 1:1000
        % Input feature vector instance
        f = x(m, :);
        % Convolution
        conved = zeros([998 1]);
        for i = 1:998
            conved(i) = sum(g.*f(i:i+2));
        end
        % Average Pooling (downsampled by 2)
        pooled = zeros([499 1]);
        for i = 1:499
            pooled(i) = mean(conved(i:i+1));
        end
        % Feedforward
        for h = 1:H1
            z1(h) = sigmoid(sum(w1(h, :)*pooled) + b1);
        end
        for h = 1:H2
            z2(h) = sigmoid(sum(w2(h, :).*z1) + b2);
        end
        y_p(m) = sigmoid(sum(w3.*z2) + b3);
    end
    % --- Back Propogation --- %
    for h = 1:H2
        del_w3(h) = -lr*(y(m) - y_p(m))*y_p(m)*(1-y_p(m))*z2(h);
    end
    del_b3 = -lr*(y(m) - y_p(m))*y_p(m)*(1-y_p(m));
    
    for h2 = 1:H2
        for h1 = 1:H1
            sigma =(y(m)-y_p(m))*w3();
        end
    end
    
    
    
end
