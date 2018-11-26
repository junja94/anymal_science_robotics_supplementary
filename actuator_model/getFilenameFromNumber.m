function [ fname ] = getFilenameFromNumber( fileNumber, fileDir )
%GETFILENAMEFROMNUMBER Summary of this function goes here
%   Detailed explanation goes here
% Add zeros in front of the number
fileNumberString = num2str(fileNumber);
for j=1:1:5-length(fileNumberString)
    fileNumberString = ['0' fileNumberString];
end

fname = '';
listingDirs = dir(fileDir);
for k=1:length(listingDirs)
    if (listingDirs(k).isdir == 0), 
        res = strfind(listingDirs(k).name, fileNumberString);
        if (length(res) > 0),
            fname = listingDirs(k).name;
            break;
        end
    end
end

if( isempty(fname) )
    msg = strcat('File with number: ', fileNumberString ,' not found in directory: ', fileDir);
    error(msg)
end

fname = [fileDir  '/' fname];

end

