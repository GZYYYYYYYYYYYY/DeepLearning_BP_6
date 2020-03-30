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
files = dir(fullfile('F:\Study\Deep_study\Lab6_1\random_animal\','*.JPEG'));%�����ͼƬ��ʽΪjpeg
lengthFiles = length(files);
%lengthFiles = 200;
A=cell(lengthFiles,1);%��cell���洢ÿ��ͼƬ����Ӧ�ľ���

for i = 1:lengthFiles;
Img = imread(strcat('F:\Study\Deep_study\Lab6_1\random_animal\',files(i).name));%�ļ�����·��
A{i,1}=Img;
J = imresize(Img, [64, 64]);
%�ж��Ƿ��ǲ�ͼ
mysize=size(J);
if numel(mysize)>2
  GImg = rgb2gray(J); %����ɫͼ��ת��Ϊ�Ҷ�ͼ��
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
files = dir(fullfile('F:\Study\Deep_study\Lab6_1\lion_train\','*.JPEG'));%�����ͼƬ��ʽΪjpeg
lengthFiles = length(files);
%lengthFiles = 200;
A=cell(lengthFiles,1);%��cell���洢ÿ��ͼƬ����Ӧ�ľ���

for i = 1:lengthFiles;
Img = imread(strcat('F:\Study\Deep_study\Lab6_1\lion_train\',files(i).name));%�ļ�����·��
A{i,1}=Img;
J = imresize(Img, [64, 64]);
%�ж��Ƿ��ǲ�ͼ
mysize=size(J);
if numel(mysize)>2
  GImg = rgb2gray(J); %����ɫͼ��ת��Ϊ�Ҷ�ͼ��
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
files = dir(fullfile('F:\Study\Deep_study\Lab6_1\lion_test\','*.JPEG'));%�����ͼƬ��ʽΪjpeg
lengthFiles = length(files);
%lengthFiles = 200;
A=cell(lengthFiles,1);%��cell���洢ÿ��ͼƬ����Ӧ�ľ���

for i = 1:lengthFiles;
Img = imread(strcat('F:\Study\Deep_study\Lab6_1\lion_test\',files(i).name));%�ļ�����·��
A{i,1}=Img;
J = imresize(Img, [64, 64]);
%�ж��Ƿ��ǲ�ͼ
mysize=size(J);
if numel(mysize)>2
  GImg = rgb2gray(J); %����ɫͼ��ת��Ϊ�Ҷ�ͼ��
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




