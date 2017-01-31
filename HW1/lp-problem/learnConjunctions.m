% This script is to help you solve the conjunction learning problem.
% You do not need to change this file.
% Run this script after
% 1- filling in hw1sample2d.txt
% 2- finishing the implementation of findLinearDiscriminant function
% 3- implementing the plot2dSeparator function

%% experiment with 2d case
data2d = readFeatures('hw1sample2d.txt',2);
[w,theta,delta] = findLinearDiscriminant(data2d);
w
theta
delta
plot2dData(data2d);
plot2dSeparator(w,theta);

%% experiment with given nd case
data = readFeatures('hw1conjunctions.txt',10);
[w,theta,delta] = findLinearDiscriminant(data);
w
theta
delta

%% save the model
outputfile = fopen('p3b2-model.txt','w');
fprintf(outputfile, '%f\n', w);
fprintf(outputfile, '%f\n', theta);
fprintf(outputfile, '%f\n', delta);
fclose(outputfile);
