% This function writes features for the badge problem.
% You do not need to change this file.

function writeBadgeFeatures(alphabet, positions, badgeFile, featureFile)

fin = fopen(badgeFile,'r');
fout = fopen(featureFile,'w');

% read the first line
tline = fgetl(fin);
while ischar(tline) % returns true if tline is char array
    label = tline(1); % extract label part
    name  = tline(3:end); % extract name part
    
    % for each character position
    for i = 1:length(positions)
        pos = positions(i);
        % for each character
        for j = 1:length(alphabet)
            x = 0;
            if pos <= length(name)
                c = alphabet(j);
                if strcmpi(name(pos),c) % if name(pos) matches c 
                %strcmpi does case insensitive comparison
                    x = 1;
                end
            end
            fprintf(fout, '%d ', x);
        end
    end
    
    if label == '+'
        y = 1;
    else
        y = -1;
    end
    fprintf(fout, '%d\n', y);
    
    % read a new line
    tline = fgetl(fin);
end

fclose(fin);
fclose(fout);

end
