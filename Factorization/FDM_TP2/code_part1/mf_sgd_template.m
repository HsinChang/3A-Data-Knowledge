clear
close all
clc

%% load the data

% The data will be in a 'sparse matrix' format
tmp = load('ml_1m.mat'); 
Xsp = tmp.X;

%visualize the non-zero elements in Xsp (if there is dot in the plot, there is a
%non-zero element)
spy(Xsp);

%we can obtain the mask matrix from Xsp
M = double(Xsp==1);

clear tmp %release the variable from the memory

[I,J] = size(Xsp);
N = nnz(Xsp); %number of non-zeros

fprintf('Only %2.4f percent of the matrix is full!\n',100*N/(I*J));

%create a list of the elements, which will be useful later
[Xi,Xj,Xs] = find(Xsp);
Xlist = [Xi,Xj,Xs];

clear Xi
clear Xj
clear Xs


%% initialize the factor matrices

K = 20; %set the rank

W = 1 * randn(I,K); 
H = 1 * randn(K,J);

%% stochastic gradient descent


batchSize = 1000; %the number of elements that we will use at each iteration

eta = 0.03; %step-size
numIter = 100;

rmse_sgd = zeros(numIter,1);

for t = 1:numIter
    
    %get a random batch from the data
    data_index = randperm(N,batchSize);%todo
    %this vector will contain some random numbers between 1 and N. 
    %Its size should be batchSize. You can use the function randperm
    
    %for each element in the data batch, update the corresponding elements
    %in W and H
    for i = 1:batchSize
       % for each element in the batch, find its corresponding 'i', 'j',
       % and value by using the Xlist array
       cur_i = Xlist(data_index(i),1);%todo
       cur_j = Xlist(data_index(i),2);%todo
       cur_x = Xlist(data_index(i),3);%todo
       
       %compute the current xhat, for the current i and j
       cur_xhat = W(cur_i,:) * H(:,cur_j);%todo
       
       %compute the gradients for the 'corresponding elements' of W and H
       %not all the elements of W and H will be updated
	     grad_w = - (cur_x - cur_xhat) * H(:, cur_j)'; %todo
       grad_h = - W(cur_i, :)' * (cur_x - cur_xhat); %todo
       
       %take a gradient step
       W(cur_i,:) = W(cur_i,:) - eta * grad_w;%todo
       H(:,cur_j) = H(:,cur_j) - eta * grad_h;%todo
    end
    
    %compute the root-mean-squared error
    rmse_sgd(t) = sqrt(sum(sum(((M.*(W*H))-Xsp).^2))/N);%todo
    
    fprintf('Iteration %d\n',t);
end

plot(rmse_sgd);
xlabel('iterations');
ylabel('error');
%ylim([3.76 3.83])
title('\eta = 0.03, batch Size:2500');

%% Make recommendations

user_index = 11;

%use W, H, and M to compute a 'movie_index' for the user
[~,movie_index] = max((1-M(:,user_index)).*(W*H(:,user_index)))%todo

fprintf('Recommend movie %d to user %d\n',movie_index,user_index);



