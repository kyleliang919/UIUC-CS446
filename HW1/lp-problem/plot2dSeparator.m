% This function plots the linear discriminant.
% YOU NEED TO IMPLEMENT THIS FUNCTION

function plot2dSeparator(w, theta)
    x1=0:0.1:1.5;
    x2= (-w(1)*x1-theta)/w(2);
    plot(x1,x2);
end
