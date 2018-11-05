% Define normalizing function
normalize = @(v) (v-mean(v))/std(v);
% Num of iterations
k = 500;
% Initialize variables
x = [ones(349,1) normalize(x1) normalize(x2)];
[m, n] = size(x);
w = [randn(1, n); zeros(k-1, n)];
h = zeros(m, 1);
cost = zeros(k, 1);
% Learning rate (alpha)
lr = 0.01;
% Start learning
for iter = 1:k
    % Update hypothesis
     h = x*w(iter,:)';
    % Update cost
    cost(iter) =(1/2*m)*((h-y)'*(h-y));
    % Update weights (Batch GD)
    for j = 1:n
        w(iter, j) = w(iter, j) - lr*(1/m)*((h-y)'*x(:,j));
    end
    w(iter+1,:) = w(iter,:);
end
% J vs k
figure(1)
plot(cost);