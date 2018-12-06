clc;
clear;

% Standardizing function
normalize = @(v) (v-mean(v))/std(v);
% ---------------------------- %

% Load data
raw = csvread('data3.csv', 1, 0);
x1 = raw(:, 1);
x2 = raw(:, 2);
x3 = raw(:, 3);
x4 = raw(:, 4);
y = raw(:, 5);

% Prepare data (60:40 hold-out)
data = [ones(100,1) normalize(x1) normalize(x2) normalize(x3) normalize(x4) y];
shuffled = data(randperm(size(data,1)),:);
x = shuffled(:,end-1);
y = shuffled(:,end);
x_train = x(1:60,:);
x_test = x(61:100,:);
y_train = y(1:60,:);
y_test = y(61:100,:);
[m, n] = size(x_train);

% Initialize variables
Pxy = zeros(40, 2);
Py = zeros(1, 2);

% Calculate Priors
for i = 1:size(y, 1)
    if y(i) == 1
        Py(1) = Py(1) + 1;
    else
        Py(2) = Py(2) + 1;
    end
end
Py(1) = 1/Py(1);
Py(2) = 1/Py(2);

% Calculate Likelihood
mu = [mean(x(y==1)) mean(x(y==2))]; 
sigma = [std(x(y==1)) std(x(y==2))];
for i = 1:40
    Pxy(i, 1) = (1/(sqrt(2*pi)*sqrt(abs(sigma(1)))))*exp(-0.5*((x_test(i)-mu(1))^2)/(sigma(1)^2));
    Pxy(i, 2) = (1/(sqrt(2*pi)*sqrt(abs(sigma(2)))))*exp(-0.5*((x_test(i)-mu(2))^2)/(sigma(2)^2));
end

% Use LRT and predict
y_p = zeros(40, 1);
for i = 1:40
    if Pxy(i, 1)*Py(1) > Pxy(i, 2)*Py(2)
        y_p(i) = 1;
    elseif Pxy(i, 1)*Py(1) < Pxy(i, 2)*Py(2)
        y_p(i) = 2;
    end
end

% Accuracy
correct = 0;
for i = 1:40
    if y_p(i) == y_test(i)
        correct = correct + 1;
    end
end
acc = correct/45;
disp(['Accuracy: ', num2str(acc)]);