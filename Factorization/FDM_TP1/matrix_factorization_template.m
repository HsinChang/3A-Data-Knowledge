%% Factorization-Based Data Modeling --  TP 1: matrix factorization - Umut Simsekli

clear 
close all
clc

%% Data generation

% the size of the matrices will be X: I x J, W: I x K, H: K x J
I = 10;
J = 20;
K = 3;

% initialize it from random numbers ("randn" can be positive or negative)
Wtrue = 2 * randn(I,K); 
Htrue = 2 * randn(K,J);

% optional: make the true factor matrices more sparse (we decide on how
% we generate the data)

randomMask1 = (rand(I,K)<0.5); %try to understand what this line is doing
randomMask2 = (rand(K,J)<0.5); %you can play with the number 0.5 and see what happens

Wtrue = Wtrue .* randomMask1;
Htrue = Htrue .* randomMask2;

% generate the data (multiply the matrices and add some more noise)
dataNoise = 1;
X = Wtrue*Htrue + dataNoise * randn(I,J);


% visualize the data with the true factor matrices

figure, %open a new figure -- google the functions that you don't know or get help from your teaching assistant

subplot(3,3,[2 3]);
imagesc(Htrue); axis xy;
colorbar;
title('True H');

subplot(3,3,[4 7]);
imagesc(Wtrue); axis xy;
colorbar;
title('True W');

subplot(3,3,[5 6 8 9]);
imagesc(X); axis xy;
colorbar;
title('X');


%% Algorithm 1: Alternating least squares (ALS)

%Initialize the factor matrices -- you can choose a better way if you have
Wals = 2 * randn(I,K);
Hals = 2 * randn(K,J);

MaxIterAls = 20;

%record the objective function values
obj_als = zeros(MaxIterAls,1);

figure, 

for i = 1:MaxIterAls
    
    Wals = % todo
    
    Hals = % todo
    
    
    Xhat = Wals * Hals;
    obj_als(i) = %todo
    
    
    % visualize the iterations
    subplot(3,3,[2 3]);
    imagesc(Hals); axis xy;
    colorbar;
    title('H ALS');

    subplot(3,3,[4 7]);
    imagesc(Wals); axis xy;
    colorbar;
    title('W ALS');

    subplot(3,3,[5 6 8 9]);
    imagesc(Xhat); axis xy;
    colorbar;
    title('Xhat');
    
    drawnow; %you need to use this function if you want to plot in a loop
    
    %pause(1); 
    
    disp(i);
    
end

figure, 
plot(obj_als);
xlabel('Iterations');
ylabel('Objective Value');
title('ALS');


%% Algorithm 2: Gradient Descent (GD)

%Initialize the factor matrices -- you can choose a better way if you have
Wgd = 2 * randn(I,K);
Hgd = 2 * randn(K,J);

MaxIterGd = 50;

%record the objective function values
obj_gd = zeros(MaxIterGd,1);

%set the step size -- and play with it to get good results
eta = 0.01;

figure, 

for i = 1:MaxIterGd
    
    %compute the gradient
    %todo
    
    %take the small step
    %todo
    
    Xhat = Wgd * Hgd;
    obj_gd(i) = %todo
    
    % visualize the iterations
    subplot(3,3,[2 3]);
    imagesc(Hgd); axis xy;
    colorbar;
    title('H GD');

    subplot(3,3,[4 7]);
    imagesc(Wgd); axis xy;
    colorbar;
    title('W GD');

    subplot(3,3,[5 6 8 9]);
    imagesc(Xhat); axis xy;
    colorbar;
    title('Xhat');
    
    drawnow; %you need to use this function if you want to plot in a loop
    
    %pause(1); 
    
    disp(i);
    
end

figure, 
plot(obj_gd);
xlabel('Iterations');
ylabel('Objective Value');
title('GD');




