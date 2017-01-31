% This function plots labeled 2d data.
% You may not need to edit this file.

function plot2dData(data)

[m, np1] = size(data);
n = np1-1;
if n~=2
    disp('only 2d data supported.')
else
    figure;
    hold on;
    grid on;
    for i = 1:m
      if data(i, 3) == -1
        plot(data(i, 1), data(i, 2), 'xg', 'MarkerSize', 20);
      else
        plot(data(i, 1), data(i, 2), 'or', 'MarkerSize', 20);
      end
    end
    % graph labels
    title('The graph');
    xlabel('The x axis');
    ylabel('The y axis');
    % adjusting axis
    axis([-0.1 1.1 -0.1 1.1]);
end

end
