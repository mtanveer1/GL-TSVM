% Please cite the following paper if you are using this code.
% Reference: Mushir Akhtar, M. Tanveer, and Mohd. Arshad. "GL-TSVM: A robust and smooth win support vector machine with guardian loss function".
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We have put a demo of the "GL-TSVM" model with the "credit_approval" dataset 

%%
clc;
clear;
warning off all;
format compact;

%% Data Preparation
addpath(genpath('C:\Users\mushi\OneDrive\Desktop\GL-TSVM-Code-for-GitHub'))
temp_data1=load('credit_approval.mat');

temp_data=temp_data1.credit_approval;
[d,e] = size(temp_data);
%define the class level +1 or -1
for i=1:d
    if temp_data(i,e)==0
        temp_data(i,e)=-1;
    end
end

X=temp_data(:,1:end-1); mean_X = mean(X,1); std_X = std(X);
X = bsxfun(@rdivide,X-repmat(mean_X,size(X,1),1),std_X);
All_Data=[X,temp_data(:,end)];

[samples,~]=size(All_Data);
split_ratio=0.8;
test_start=floor(split_ratio*samples);
training_Data = All_Data(1:test_start-1,:); testing_Data = All_Data(test_start:end,:);
test_x=testing_Data(:,1:end-1); test_y=testing_Data(:,end);
train_x=training_Data(:,1:end-1); train_y=training_Data(:,end);



%% %% Hyperparameter range
% C=10.^[-6:2:6];
% c=10.^[-6:2:6];
% sigma=10.^[-6:2:6];
% a= 0.1:0.2:5.1;
%%
C=10^-6; % Regularization parameters
c=1;
sigma=100; % Kernel parameter
a= 0.1;   % loss function parameter



[Accuracy]=GL_TSVM_function(train_x,train_y,test_x,test_y,a,C,c,sigma);


fprintf(1, 'Testing Accuracy of GL-TSVM model is: %f\n', Accuracy);
