clear;
clc;
pkg load image;
%% Step 1: Data Preparation 
 
% loading dataset

% trainData: a matrix with size of 28x28x10000   
% trainLabels: a matrix with size of 10x10000
% testData: a matrix with size of 28x28x2000
% testLabels: a matrix with size of 10x2000

% new
% trainData: a matrix with size of 64x64x7800   
% trainLabels: a matrix with size of 64x64x7800
% testData: a matrix with size of 64x64x160
% testLabels: a matrix with size of 64x64x40



% 
% 读取图片并且转化为所需矩阵
% 
image = dir('F:\Study\Deep_study\Lab6_1\random_animal\');
files = dir(fullfile('F:\Study\Deep_study\Lab6_1\random_animal\','*.JPEG'));%处理的图片格式为jpeg
lengthFiles = length(files);
%lengthFiles = 200;
A=cell(lengthFiles,1);%用cell来存储每个图片所对应的矩阵

for i = 1:lengthFiles;
Img = imread(strcat('F:\Study\Deep_study\Lab6_1\random_animal\',files(i).name));%文件所在路径
A{i,1}=Img;
J = imresize(Img, [64, 64]);
%判断是否是彩图
mysize=size(J);
if numel(mysize)>2
  GImg = rgb2gray(J); %将彩色图像转换为灰度图像
else
GImg=J;
end

K{i} = GImg; 
%X_train{i}=K{:,:,i};
X_train_cell{i} = reshape(K{1,i},[4096,1]);
for j = 1:4096
  trainData_cell{j,i} = X_train_cell{1,i}(j);
end
end


image = dir('F:\Study\Deep_study\Lab6_1\lion_train\');
files = dir(fullfile('F:\Study\Deep_study\Lab6_1\lion_train\','*.JPEG'));%处理的图片格式为jpeg
lengthFiles = length(files);
%lengthFiles = 200;
A=cell(lengthFiles,1);%用cell来存储每个图片所对应的矩阵

for i = 1:lengthFiles;
Img = imread(strcat('F:\Study\Deep_study\Lab6_1\lion_train\',files(i).name));%文件所在路径
A{i,1}=Img;
J = imresize(Img, [64, 64]);
%判断是否是彩图
mysize=size(J);
if numel(mysize)>2
  GImg = rgb2gray(J); %将彩色图像转换为灰度图像
else
GImg=J;
end

K{i} = GImg; 
%X_train{i}=K{:,:,i};
X_train_cell{i} = reshape(K{1,i},[4096,1]);
for j = 1:4096
  testData_cell{j,i} = X_train_cell{1,i}(j);
end

end

image = dir('F:\Study\Deep_study\Lab6_1\lion_test\');
files = dir(fullfile('F:\Study\Deep_study\Lab6_1\lion_test\','*.JPEG'));%处理的图片格式为jpeg
lengthFiles = length(files);
%lengthFiles = 200;
A=cell(lengthFiles,1);%用cell来存储每个图片所对应的矩阵

for i = 1:lengthFiles;
Img = imread(strcat('F:\Study\Deep_study\Lab6_1\lion_test\',files(i).name));%文件所在路径
A{i,1}=Img;
J = imresize(Img, [64, 64]);
%判断是否是彩图
mysize=size(J);
if numel(mysize)>2
  GImg = rgb2gray(J); %将彩色图像转换为灰度图像
else
GImg=J;
end

K{i} = GImg; 
%X_train{i}=K{:,:,i};
X_train_cell{i} = reshape(K{1,i},[4096,1]);
for j = 1:4096
  testLabels_cell{j,i} = X_train_cell{1,i}(j);
end
%testData(:,i) = reshape(X_train{1,i},[4096,1]);
end

train_size = 7800; % number of training samples
% input in the 1st layer 


trainData_mat=cell2mat(trainData_cell);
trainData=im2double(trainData_mat);

trainLabels=trainData;

testData_mat=cell2mat(testData_cell);
testData=im2double(testData_mat);

testLabels_mat=cell2mat(testLabels_cell);
testLabels=im2double(testLabels_mat);

X_train = reshape(trainData, 4096, train_size);
%X_train_mat=cell2mat(X_train_cell);
%X_train=im2double(X_train_mat);

test_size = 160; % number of testing samples
% external input in the 1st layer 
X_test = reshape(testData, 4096, test_size); 

%% Step 2: Design Network Architecture
% define number of layers
L = 5; 
% define number of neurons in each layer 
layer_size = [4096 % number of neurons in 1st layer
              1024 % number of neurons in 2nd layer
              256 % number of neurons in 3rd layer
              1024  % number of neurons in 4th layer
              4096];% number of neurons in 5th layer
L2 = 3;
layer2_size = [4096 % number of neurons in 1st layer
              1024 % number of neurons in 2nd layer
              256];% number of neurons in 3th layer   
%% Step 3: Initial Parameters

% initialize weights in each layer with Gaussian distribution
for l = 1:L-1
    w{l} = 0.1 * randn(layer_size(l+1,1), sum(layer_size(l,:)));
end
 
alpha = 0.005; % initialize learning rate 
beta = 0.01;

%% Step 4: Define Cost Function
% cost function is defined in cost.m

%% Step 5: Define Evaluation Index
% accuracy defined in accuracy.m

%% Step 6: Train the Network
J = []; % array to store cost of each mini batch
Acc = []; % array to store accuracy of each mini batch
max_epoch = 200; % number of training epoch
mini_batch = 78; % number of sample of each mini batch

figure % plot the cost
for iter=1:max_epoch
    % randomly permute the indexes of samples in training set
idxs = randperm(train_size); 
% for each mini-batch
for k = 1:ceil(train_size/mini_batch)
    % prepare internal inputs in 1st layer denoted by a{1}
    
    start_idx = (k-1)*mini_batch+1;          % start index of kth mini-batch
    end_idx = min(k*mini_batch, train_size); % end index of kth mini-batch
	a{1} = X_train(:,idxs(start_idx:end_idx));
    % prepare labels
    y = trainLabels(:, idxs(start_idx:end_idx));
    
        % forward computation
        for l=1:L-1
            [a{l+1}, z{l+1}] = fc(w{l}, a{l});
        end
 
        % Compute delta of last layer
        delta{L} = (a{L} - y).* a{L} .*(1-a{L}); %delta{L}={partial J}/{partial z^L} 
 
        % backward computation
        for l=L-1:-1:2
            delta{l} = bc(w{l}, z{l}, delta{l+1}, beta);
        end
 
        % update weight 
        for l=1:L-1
            % compute the gradient
            grad_w = delta{l+1} * a{l}'; 
            w{l} = w{l} - alpha*grad_w;
        end 
        %betasuma = ones(4096,78);
##        for l=1:L-1
##            betasuma += a{l};
##        end
        %betasuma = beta*betasuma;
        % training cost on training batch
        J = [J 1/mini_batch*sum(cost(a{L}, y, beta))];
        Acc =[Acc accuracy(a{L}, y)]; 
        % plot training error 
        plot(J);
        pause(0.000001);
    end
    
    
    for j=1:160
      b{1} = testData(:,j);
    for l=1:L2-1
        [b{l+1}, h{l+1}] = fc(w{l}, b{l});
    end
    end
end

%new train
for l = 1:L2-1
    w{l} = 0.1 * randn(layer_size(l+1,1), sum(layer_size(l,:)));
end

alpha = 0.005; % initialize learning rate 

%% Step 4: Define Cost Function
% cost function is defined in cost.m

%% Step 5: Define Evaluation Index
% accuracy defined in accuracy.m

%% Step 6: Train the Network
J = []; % array to store cost of each mini batch
Acc = []; % array to store accuracy of each mini batch
max_epoch = 10;
mini_batch = 16;
for iter=1:max_epoch
    % randomly permute the indexes of samples in training set
idxs = randperm(test_size); 
% for each mini-batch
for k = 1:ceil(test_size/mini_batch)  %10
    % prepare internal inputs in 1st layer denoted by a{1}
    
    start_idx = (k-1)*mini_batch+1;          % start index of kth mini-batch
    end_idx = min(k*mini_batch, test_size); % end index of kth mini-batch
	a{1} = b{L2};
    % prepare labels
    y = testLabels(:, idxs(start_idx:end_idx));
    
        % forward computation
        for l=1:L-1
            [a{l+1}, z{l+1}] = fc(w{l}, a{l});
        end
 
        % Compute delta of last layer
        delta{L} = (a{L} - y).* a{L} .*(1-a{L}); %delta{L}={partial J}/{partial z^L} 
 
        % backward computation
        for l=L-1:-1:2
            delta{l} = bc(w{l}, z{l}, delta{l+1}, 0);
        end
 
        % update weight 
        for l=1:L-1
            % compute the gradient
            grad_w = delta{l+1} * a{l}'; 
            w{l} = w{l} - alpha*grad_w;
        end 

        % training cost on training batch
        J = [J 1/mini_batch*sum(cost(a{L}, y, 0))];
        Acc =[Acc accuracy(a{L}, y)]; 
        % plot training error 
        plot(J);
        pause(0.000001);
    end
end 

% end training
% plot accuracy
figure
plot(Acc);

%% Step 7: Test the Network
%test on training set
a{1} = X_train;
for l = 1:L-1
  a{l+1} = fc(w{l}, a{l});
end
train_acc = accuracy(a{L}, trainLabels);
fprintf('Accuracy on training dataset is %f%%\n', train_acc*100);

%test on testing set
a{1} = X_test;
for l = 1:L-1
   a{l+1} = fc(w{l}, a{l});
end
test_acc = accuracy(a{L}, testLabels);
fprintf('Accuracy on testing dataset is %f%%\n', test_acc*100);

%% Step 8: Store the Network Parameters
save model.mat w layer_size

