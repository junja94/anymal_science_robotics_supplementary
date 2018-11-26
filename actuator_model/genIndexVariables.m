function [ ] = genIndexVariables( logElements , verbose)
%GENINDEXVARIABLES Summary of this function goes here
%   Detailed explanation goes here
% % generate index variables

if nargin < 2
 verbose = true;
end

for k=1:length(logElements)
    completeName = char(logElements(k).name);
    completeName = completeName(5:end);
    [startIndex,endIndex] = regexp(completeName ,'/');
    
    if ~isempty(endIndex)
        strippedName = completeName(endIndex(end)+1:end);
    else
        strippedName = completeName;
    end
    
    % hack
    strippedName = strrep(completeName, '/', '_');
    idxName = ['idx' strippedName];
    if(verbose)
        disp([num2str(k) ' ' idxName]);
    end
    evalin('base', [' global ' idxName]);
    evalin('base', [idxName '=' num2str(k) ';']);
end

end

