clc;
clear;

x = [0 0 1 1; 0 1 0 1];
y = [0 1 1 0];
y1 = [0 0 1 0];
y2 = [0 1 0 0];
w1 = [0 0;0 0];
w2 = [0 0];
b = [0 0];
alpha = 0.1;
theta = 0.5;
iter = 50;
for k = 1:iter
     for i = 1:size(x, 2)
        p(i) = b(1) + w1(1, 1)*x(1,i) + w1(1, 2)*x(2,i);
        q(i) = b(1) + w1(2, 1)*x(1,i) + w1(2, 2)*x(2,i);
        yp1(i) = (p(i) >= theta);
        yp2(i) = (q(i) >= theta);
        if (yp1(i) ~= y1(i))
            for j = 1:2
                w1(1, j) = w1(1, j) + alpha*y1(i)*x(j,i);
            end
            b(1) = b(1) + alpha*y1(i);
        end
        if (yp2(i) ~= y2(i))
            for j = 1:2
            w1(2,j) = w1(2,j) + alpha*y2(i)*x(j,i);
            end
            b(1) = b(1) + alpha*y2(i);
        end
        r(i) = b(2) + w2(1)*y1(i) + w2(2)*y2(i);
        h(i) = (r(i) >= theta);
        if (h(i) ~= y(i))
            for j = 1:2
                w2(j) = w2(j) + alpha*y(i)*y1(i);
            end
            b(2) = b(2) + alpha*y(i);
        end
     end
     e(k) = sum((y-h).^2);
end
plot(e)