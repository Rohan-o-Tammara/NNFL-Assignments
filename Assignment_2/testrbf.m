clc;
clear;

data = xlsread('dataset.xlsx');
X = data(:,(1:7));
Y = data(:,8);
z = zeros(150,3);
% No. of output neurons = 3

for i = 1:length(Y)
    if (Y(i) == 1)
        z(i,:) = [1,0,0];
    elseif (Y(i) == 2)
        z(i,:) = [0,1,0];
    elseif (Y(i) == 3)
        z(i,:) = [0,0,1];
    end
end

% Dividing data into test and training : 70-30 cross validation
traininput = [];
trainoutput = [];
testinput = [];
testoutput = [];

for j = 1:size(X,1)
    if rand < 0.7
        traininput = [traininput; X(j,:)];
        trainoutput = [trainoutput;z(j,:)];
    else
        testinput = [testinput; X(j,:)];
        testoutput = [testoutput;z(j,:)];
    end
end

x = traininput;
y = trainoutput;
xt = testinput;
yt = testoutput;

[~ , mu] = kmeans(x,10);
%[~ , mu] = kmeans(x,n);

% Hidden layer eval
for i = 1:size(x,1)
    for j = 1:size(mu,1)
        h(i,j) = (norm( x(i,:) - mu(j,:)))^3;
    end
end

% Weight eval
W = pinv(h)*y;

% Test data eval

for i1 = 1:size(xt,1)
    for j1 = 1:size(mu,1)
        H(i1,j1) = (norm( xt(i1,:) - mu(j1,:)))^3;
    end
end
output = H*W;
pl = zeros(1,40);
pa = zeros(1,40);

for i1 = 1:size(output,1)
    [~,pl(i1)] = max(output(i1,:));
    [~,pa(i1)] = max(yt(i1,:));
end

[cm,~] = confusionmat(pa,pl);

% fprintf("The confusion matrix is : \n");
 disp(cm)
