% This function computes the accuracy of the classifier (linear separator
% represented by weights w and threshold theta) in the given labeled data.
% You do not need to change this file.

function accuracy = computeAccuracy(data, w, theta)

[m, np] = size(data);
numCorrect = 0.0;
for i=1:m
    x = data(i,1:end-1)';
    yTrue = data(i,end);
    yPredicted = computeLabel(x, w, theta);
    if yTrue==yPredicted
        numCorrect = numCorrect + 1.0;
    end
end
accuracy = numCorrect/m;

end
