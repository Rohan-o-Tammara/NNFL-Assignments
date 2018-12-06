clc;
clear;

x = [0 0 1 1; 0 1 0 1];
y = [0 1 1 0];
w = [0 0];
b = [0 0];
alpha = 0.1;
theta = 0.5;
iter = 50;
for k = 1:iter
    for i=1:size(x,2)
        p(i) = b(1) + x(1,i)*w(1) + x(2,i)*w(2);
        q(i) = b(1) + x(1,i)*(1-w(1)) + x(2,i)*(1-w(2));
        y1 = (p(i) >= theta);
        y2 = (q(i) >= theta);
        r(i) = b(2) + y1 + y2;
        h(i) = (r(i) >= theta);
        if (h(i) ~= y(i))
            for j=1:2
                w(j) = w(j) + (alpha*(y1+y2)*x(j,i));
            end
            b(2) = b(2) + (alpha*y(i));
            b(1) = b(1) + alpha*(y1+y2);
        end
    end
    e(k) = sum((y-h).^2);
end
plot(e)