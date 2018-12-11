function w = logistic_regression( x_train, y_train, k, lr, w)
% ----------------------------------------------------------------- %
% Funtion to perform logistic regression on data and return weights %
% ----------------------------------------------------------------- %
    [m, n] = size(x_train);
    % Initialize variables
    if ~exist('w', 'var')
        w = rand(1, n);
    end
    h = zeros(m, 1);
    % Start training
    for iter = 1:k
        % Update hypothesis
        for i = 1:m
            h(i) = 1/(1 + exp(-(x_train(i,:)*w')));
        end
        % Update cost
        for i = 1:m
            cost = (y_train(i)*log(h(i)) + (1-y_train(i))*log(1-h(i)));
        end
        % Update weights
        for j = 1:n
            gradient = 0;
            for i = 1:m
                gradient = gradient + (y_train(i)*(1-h(i))+(1-y_train(i))'*h(i))*x_train((i),j);
            end
            w(j) = w(j) - lr*gradient;
        end
    end
end

