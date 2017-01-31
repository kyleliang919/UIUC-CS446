% This function solves the LP problem for a given weight vector
% to find the threshold theta.
% YOU NEED TO FINISH IMPLEMENTATION OF THIS FUNCTION.

function [theta,delta] = findLinearThreshold(data,w)
%% setup linear program
[m, np1] = size(data);
n = np1-1;
% write your code here
% c'*t=delta,which is what we want to minimize,given t=[w1,w2,...wn,theta,delta]
c=zeros(n+2,1);
c(n+2,1)=1;
% y(w'x+theta)+delta>=1 ==> w1*y*x1+w2*y*x2+...+wn*y*xn+y*theta+delta>=1
% diag() creates a matrix that puts all the y on diagonal
A=[[diag(data(:,np1))*data(:,1:n),data(:,np1);zeros(1,np1)],ones(m+1,1)];
% b is just a vector of m ones, since y(w'x+theta)+delta>=1; the zero last
% element is for theta>=0
b=ones(m+1,1);
b(m+1,1)=0;
%% solve the linear program
%adjust for matlab input: A*x <= b
[t, z] = linprog(c, -A, -b, [], [], [w' -inf -inf], [w' inf inf]);
%% obtain w,theta,delta from t vector
w = t(1:n);
theta = t(n+1);
delta = t(n+2);
end
