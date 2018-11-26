function [ leOut ] = matchTimeToData( leIn, timeSyncOffset, sTimeIdx, nsTimeIdx)
%MATCHTIMETODATA Summary of this function goes here
%   Detailed explanation goes here

% Set default indices of seconds and nanoseconds
if nargin < 4
 nsTimeIdx = 2;
end
if nargin < 3
 sTimeIdx = 1;
end

% Set out equals in
leOut = leIn;

% Check buffer type
for i=1:length(leOut)
    startIdx = 1;
    stopIdx = leOut(i).noData * leOut(i).divider;
    if(leOut(i).isBufferLooping)
        startIdx = leOut(sTimeIdx).noData - mod((timeSyncOffset - 1), leOut(i).divider) ...
                    - (leOut(i).noData - 1) * leOut(i).divider;
        stopIdx = startIdx + leOut(i).divider*(leOut(i).noData - 1);
    end
    
    leOut(i).timeStruct.seconds = leOut(sTimeIdx).data(startIdx:leOut(i).divider:stopIdx);
    leOut(i).timeStruct.nanoseconds = leOut(nsTimeIdx).data(startIdx:leOut(i).divider:stopIdx);
    leOut(i).systime = double(leOut(i).timeStruct.seconds) + double(leOut(i).timeStruct.nanoseconds)*1.0e-9;
    leOut(i).time = leOut(i).systime - leOut(i).systime(1);
end

end

