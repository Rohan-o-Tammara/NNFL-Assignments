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
x_train = data(1:700, 1:1000);
y_train = data(1:700, 1001);
x_test = data(701:1000, 1:1000);
y_test = data(701:1000, 1001);
[M, N] = size(x_train);
[P, Q] = size(x_test);
Nc = N-2;
Np = (N-2)/2;

% No. of hidden neurons
H1 = 40;
H2 = 20; %

% Learning rate
lr = 0.1;
glr = 1e-8;

% Number of iterations
iterations = 5;

% Initialize Kernel
g = [1/3 1/3 1/3];

% Number of classes
K = 1;

% Initialize weights and bias(random value between -0.01 to 0.01)
w1 = (rand([H1 Np]) - rand([H1 Np]))/100;
w2 = (rand([H2 H1]) - rand([H2 H1]))/100;
w3 = (rand([K H2]) - rand([K H2]))/100;
b1 = (rand - rand)/100;
b2 = (rand - rand)/100;
b3 = (rand - rand)/100;

% Initialize intermediates (for our ease)
z1 = zeros([1 H1]);
z2 = zeros([1 H2]);
y = zeros([M K]);
del_w1 = zeros([H1 N]);
del_w2 = zeros([H2 H1]);
del_w3 = zeros([K H2]);

cost = zeros([iterations 1]);
y_p = zeros([P K]);

% Start Training
for k = 1:iterations
    for m = 1:M
       cost(k) = cost(k) + (y(m) - y_train(m)).^2;
    end
    cost(k) = 0.5*sqrt(cost(k));
    % ---- Forward Propogation --- %
    for m = 1:M
        % Input feature vector instance
        f = x_train(m, :);
        % Convolution
        conved = zeros([Nc 1]);
        for i = 1:Nc
            conved(i) = g*f(i:i+2)';
        end
        % Average Pooling (downsampled by 2)
        pooled = zeros([Np 1]);
        for i = 1:Np
            pooled(i) = mean(conved(i:i+1));
        end
        % Feedforward
        for h = 1:H1
            z1(h) = sigmoid(w1(h, :)*pooled + b1);
        end
        for h = 1:H2
            z2(h) = sigmoid(sum(w2(h, :).*z1) + b2);
        end
        for i = 1:K
            y(m, i) = sigmoid(sum(w3(i, :).*z2) + b3);
        end
    % --- Back Propogation --- %
        % Update Weights and Biases
        for i = 1:K
            for h = 1:H2
                del_w3(i, h) = -lr*(y_train(m, i)-y(m, i))*y(m, i)*(1-y(m,i))*z2(h);
            end
            del_b3 = -lr*(y_train(m, i)-y(m, i))*y(m, i)*(1-y(m,i));
        end
        
        for h2 = 1:H2
            for h1 = 1:H1
               sigma = 0;
               for i = 1:K
                   sigma = sigma + (y_train(m, i)-y(m,i))*w3(i,h2);
               end
               del_w2(h2, h1) = -lr*sigma*z2(h2)*(1-z2(h2))*z1(1,h1);
            end
            del_b2 = -lr*sigma*z2(h2)*(1-z2(h2));
        end
        
        for h1 = 1:H1
            for j = 1:Np
               sigma = 0;
               for h2 = 1:H2
                   for i = 1:K
                       sigma = sigma + (y_train(m,i)-y(m,i))*w3(i,h2)*w2(h2,h1);
                   end
               end
               del_w1(h1,j) = -lr*sigma*z1(h1)*(1-z1(h1))*pooled(j);
            end
            del_b1 = -lr*sigma*z1(h1)*(1-z1(h1));
        end
        
        for i = 1:K
            for h = 1:H2
                w3(i,h) = w3(i,h) - del_w3(i,h);
            end
            b3 = b3 - del_b3;
        end
        
        for h2 = 1:H2
            for h1 = 1:H1
                w2(h2,h1) = w2(h2,h1) - del_w2(h2,h1);
            end
            b2 = b2 - del_b2;
        end
        
        for h = 1:H1
            for j = 1:Np
                w1(h,j) = w1(h,j) - del_w1(h,j);
            end
            b1 = b1 - del_b1;
        end
        
        
        % Upsample
        upsampled = zeros([Nc 1]);
        for i = 1:2:Nc
            upsampled(i:i+1) = pooled((i+1)/2);
        end
        upsampled = [upsampled;0;0];
        
        %{
        % Update Kernel
        del_g = 0;
        for i = 1:3
            delta = 0;
            for j = 1:Nc
                delta = delta + f(j:j+2)*upsampled(j:j+2);
            end
            del_g = del_g + delta;
        end
        g = g - glr*del_g;          
    %}
    end
    disp(['Iteration: ', num2str(k)]);
end
% --- Validation --- %
for p = 1:P     
    ft = x_test(p, :);
    convedt = zeros([Nc 1]);
    for i = 1:Nc
        convedt(i) = g*ft(i:i+2)';
    end
    pooledt = zeros([Np 1]);
    for i = 1:Np
        pooledt(i) = mean(convedt(i:i+1));
    end
    for h = 1:H1
        z1t(h) = sigmoid(sum(w1(h, :).*pooledt(p, :)) + b1);
    end
    for h = 1:H2
        z2t(h) = sigmoid(sum(w2(h, :).*z1t) + b2);
    end
    for i = 1:K
        y_p(p, i) = sigmoid(sum(w3(i, :).*z2t) + b3);
        if y_p(p,i) > 0.5
            y_p(p,i) = 1;
        else
            y_p(p,i) = 0;
        end
    end
end
positive = 0;
for p = 1:P
    if y_p(p) == y_test(p)
        positive = positive + 1;
    end
end
val_acc = positive/P;
disp(['Validation Accuracy: ', num2str(val_acc)]);