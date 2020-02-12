clear
close all
clc


%% data size
s1 = 3883;
s2 = 6040;
s3 = 10;

%% Collect the rmse computations

outpath = './rmse';

rmse = [];
files = dir('./rmse/*.txt');

for i = 1:length(files)
    tmpRmse = load(['./rmse/' files(i).name]);
    rmse = [rmse;tmpRmse(:)];
end

plot(rmse);

ylabel('RMSE')
xlabel('Iteration');
title('K=10, \eta=0.0001');



