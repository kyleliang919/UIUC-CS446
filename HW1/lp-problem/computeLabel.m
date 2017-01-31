% This function computes the label for the given
% feature vector x using the linear separator represented by 
% the weight vector w and the threshold theta.
% YOU NEED TO WRITE THIS FUNCTION.

function y = computeLabel(x, w, theta)
    y=sign(x'*w+theta);
    y(y==0)=1;
end
