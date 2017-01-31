% This script is to help you solve the badge learning problem.
% You may only need to change the first two lines of this file.
% Run this script after
% 1- finishing the implementation of findLinearDiscriminant function
% 2- implementing computeLabel function

%% define the characters of interest (alphabet) and character positions of interest (positions)
alphabet = 'BCDEFGHIJKLMNOPQRSTUVWXYZ_.- ';
positions = 1:10;

%% compute features and save them to file
writeBadgeFeatures(alphabet,positions,'badges.train','features.train');
writeBadgeFeatures(alphabet,positions,'badges.test','features.test');

%% read features from file
nFeatures = length(alphabet)*length(positions);
trainData = readFeatures('features.train',nFeatures);
testData = readFeatures('features.test',nFeatures);

%% find linear discriminant using LP
[w,theta,delta] = findLinearDiscriminant(trainData);

%% compute accuracy in both training and test data
accuracyInTrain = computeAccuracy(trainData,w,theta);
accuracyInTest = computeAccuracy(testData,w,theta);

%% display results
delta
accuracyInTrain
accuracyInTest

%% plot weight vector
features = reshape(w,length(alphabet),length(positions))';
figure;
pixelmap = length(colormap)*(features-min(w))/(max(w)-min(w));
image(pixelmap);
xlabel('Alphabet');
ylabel('Positions');
set(gca,'XTick',1:length(alphabet));
set(gca,'XTickLabel',alphabet');
set(gca,'YTick',1:length(positions));
set(gca,'YTickLabel',positions);
colorbar('YTick',[1,length(colormap)],'YTickLabel', {num2str(min(w)),num2str(max(w))});
