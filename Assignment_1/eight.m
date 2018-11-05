% Standardizing function
norm = @(v) (v-mean(v))/std(v);

data = [ones(150,1) norm(x1) norm(x2) norm(x3) norm(x4) y];
shuffled = data(randperm(size(data,1)),:);
x = shuffled(:,1:5);
y = shuffled(:,6);
x_train = x(1:60,:);
x_test = x(61:100,:);
y_train = y(1:60,:);
y_test = y(61:100,:);

w_m1 = logistic_regression(x_train, y_train, 100, 0.1);
w_m2 = logistic_regression(x_train, y_train, 100, 0.1);
w_m3 = logistic_regression(x_train, y_train, 100, 0.1);