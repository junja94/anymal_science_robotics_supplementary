function [ fid, currentLine ] = skipComments( fid, resetPosition, commentString  )

if nargin < 3
 commentString = '#';
end

% Get position in file
positionInFile = ftell(fid);
% Get current line
currentLine = strtrim(fgets(fid));

% While not end of file and empty line or comment
while( ischar(currentLine) && ( isempty(currentLine) || ...
       ( ge(length(currentLine),length(commentString)) && ...
         strcmp(currentLine(1:length(commentString)), commentString) ) ) )
    % Get position in file
    positionInFile = ftell(fid);
    % Get current line
    currentLine = strtrim(fgets(fid));
end

% Reset file pointer
if(resetPosition)
    fseek(fid, positionInFile, 'bof');
end

end