% Use this file to answer b.4 of the LP problem.
% You do not need to edit this file.
% You need to implement the function findLinearDiscriminant,
% and findLinearThreshold before running this script.

data2d = [
    0.3300    0.8737 1
    0.6175    0.9614 1
    0.5014    0.6877 1
    0.6258    0.8596 1
    0.9051    0.9474 1
    0.7198    0.8632 1
    0.7088    0.6737 1
    0.9023    0.4842 1
    0.9521    0.6807 1
    1.0461    0.6632 1
    0.0728    0.6596 -1
    0.3382    0.4702 -1
    0.1088    0.3228 -1
    0.0590    0.1368 -1
    0.5069    0.2491 -1
    0.3217    0.0596 -1
    0.3106    0.2596 -1
    0.5346    0.0912 -1
    0.7945    0.1018 -1
    0.8691    0.0175 -1
    ];

plot2dData(data2d);

outputfile = fopen('p3b4-model.txt','w');
[w,theta,delta] = findLinearDiscriminant(data2d);
plot2dSeparator(w,theta);
w
theta
delta

fprintf(outputfile, '%f\n', w);
fprintf(outputfile, '%f\n', theta);
fprintf(outputfile, '%f\n', delta);

w = [100.0, 300.0]';
[theta,delta] = findLinearThreshold(data2d,w);
plot2dSeparator(w,theta);
theta
delta

fprintf(outputfile, '%f\n', theta);
fprintf(outputfile, '%f\n', delta);

w = [130.0, 100.0]';
[theta,delta] = findLinearThreshold(data2d,w);
plot2dSeparator(w,theta);
theta
delta

fprintf(outputfile, '%f\n', theta);
fprintf(outputfile, '%f\n', delta);

w = [500.0, 100.0]';
[theta,delta] = findLinearThreshold(data2d,w);
plot2dSeparator(w,theta);
theta
delta

fprintf(outputfile, '%f\n', theta);
fprintf(outputfile, '%f\n', delta);
fclose(outputfile);
