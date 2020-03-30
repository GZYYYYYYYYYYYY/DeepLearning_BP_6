##cd('F:\Study\Deep_study\Lab6_1\random_animal');
##file=dir('F:\Study\Deep_study\Lab6_1\random_animal\*.jpg');
##[k len]=size(file);
##for i=1:k
##    name=file(i).name;
##    I=imread(name);
##    fprintf(I);
##    figure(i);
##    imshow(I);
##end



clc;clear all;
pkg load image;
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
X_train{i} = reshape(K{1,i},[4096,1]);
for j = 1:4096
  trainData{j,i} = X_train{1,i}(j);
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
X_train{i} = reshape(K{1,i},[4096,1]);
for j = 1:4096
  testData{j,i} = X_train{1,i}(j);
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
X_train{i} = reshape(K{1,i},[4096,1]);
for j = 1:4096
  testLabels{j,i} = X_train{1,i}(j);
end
%testData(:,i) = reshape(X_train{1,i},[4096,1]);
end




