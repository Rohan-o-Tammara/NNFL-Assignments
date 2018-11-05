x = [0 1];
y = [1 0];
w = 0;
b = 0;
alpha = 0.1;
theta = 0.5;
iter = 50;
for k=1:iter
    for i=1:2
        a(i) = b + w*x(i);
        h(i) = (a(i) >= theta);
        if h(i) ~= y(i)
            w = w + alpha*y(i)*x(i);
            b = b + alpha*y(i);
        end
    end
    e(k) = sum((y-h).^2);
end
plot(e)

                