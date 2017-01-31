% This function reads features from a file into a matrix called data.
% You do not need to change this file.
% Each line in the featureFile must be a binary sequence followed by either
% 1 to indicate a positive label, or
% -1 to indicate a negative label.

function data = readFeatures(featureFile, nFeatures)
    fid = fopen(featureFile, 'r');
    data = fscanf(fid, '%d', [nFeatures + 1, inf])';
    fclose(fid);
end
