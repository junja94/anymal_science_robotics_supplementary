clear
clc
close all

addpath('helpers')

fNumber = 1038; %% highspeed

fName = getFilenameFromNumber(fNumber, 'data');
fprintf(['\nGot filename: ', fName, ' from number: ', num2str(fNumber)]);

% Read data
logElements = loadLogFile(fName);
fprintf(['\n\nLoaded data from binary file:', fName]);

% Generate Index Variables
verbose = false;
genIndexVariables(logElements, verbose);
fprintf('\n\nGenerated indices for the log elements!\n');

% Increase precision in data cursor and show index of data point
set(0,'defaultFigureCreateFcn',@(s,e)datacursorextra(s))

%%%%%%%%%%%%%%% CONFIG %%%%%%%%%%%%%%%%%%%%%%%%%%
time = double(logElements(1).data - logElements(1).data(1));
time = time + double(logElements(2).data) * 10^-9;

joint_idx = [6];
time_start = -0.05;
time_end = 1.03;

time = time - 3.3;
time = time - 1.3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% figure
% hold on
idx_vel = [idx_model_meas_jointVel_LF_HAA, ...
           idx_model_meas_jointVel_LF_HFE, ...
           idx_model_meas_jointVel_LF_KFE, ...
           idx_model_meas_jointVel_LH_HAA, ...
           idx_model_meas_jointVel_LH_HFE, ...
           idx_model_meas_jointVel_LH_KFE, ...
           idx_model_meas_jointVel_RF_HAA, ...
           idx_model_meas_jointVel_RF_HFE, ...
           idx_model_meas_jointVel_RF_KFE, ...
           idx_model_meas_jointVel_RH_HAA, ...
           idx_model_meas_jointVel_RH_HFE, ...
           idx_model_meas_jointVel_RH_KFE];

% for i=1:12 
%         plot(logElements(idx_vel(i)).data)
%     hold on
% end
% title('velocity')

% figure
% hold on
idx_pos = [idx_model_meas_jointPos_LF_HAA, ...
           idx_model_meas_jointPos_LF_HFE, ...
           idx_model_meas_jointPos_LF_KFE, ...
           idx_model_meas_jointPos_LH_HAA, ...
           idx_model_meas_jointPos_LH_HFE, ...
           idx_model_meas_jointPos_LH_KFE, ...
           idx_model_meas_jointPos_RF_HAA, ...
           idx_model_meas_jointPos_RF_HFE, ...
           idx_model_meas_jointPos_RF_KFE, ...
           idx_model_meas_jointPos_RH_HAA, ...
           idx_model_meas_jointPos_RH_HFE, ...
           idx_model_meas_jointPos_RH_KFE];

% for i=1:12    
%     plot(logElements(idx_pos(i)).data)
%     hold on
% end
% title('joint position')

idx_com = [ idx_command_desJointPos_LF_HAA, ...
            idx_command_desJointPos_LF_HFE, ...
            idx_command_desJointPos_LF_KFE, ...
            idx_command_desJointPos_LH_HAA, ...
            idx_command_desJointPos_LH_HFE, ...
            idx_command_desJointPos_LH_KFE, ...
            idx_command_desJointPos_RF_HAA, ...
            idx_command_desJointPos_RF_HFE, ...
            idx_command_desJointPos_RF_KFE, ...
            idx_command_desJointPos_RH_HAA, ...
            idx_command_desJointPos_RH_HFE, ...
            idx_command_desJointPos_RH_KFE];
    
for i=1:length(joint_idx) 
    
    figure('rend','painters','pos',[100 10 600 280])
set(gca,'FontSize',10) 
set(gca,'Units','pixels') 
set(gca, 'Position', [50 50 550 230])
        plot(time, logElements(idx_com(joint_idx(i))).data)
    hold on
    axis([time_start time_end 0.07 1.34])
    xlabel("Time [s]", 'FontSize', 10)
    ylabel("Desired joint position [rad]", 'FontSize', 10)

end

% legend('Desired joint position')


% figure
idx_ori = [ idx_model_meas_orientBaseToWorldQuat_w, ...
            idx_model_meas_orientBaseToWorldQuat_x,...
            idx_model_meas_orientBaseToWorldQuat_y,...
            idx_model_meas_orientBaseToWorldQuat_z];

idx_lve = [ idx_model_meas_linVelBaseInWorldFrame_x, ...
            idx_model_meas_linVelBaseInWorldFrame_y,...
            idx_model_meas_angVelBaseInBaseFrame_z];

bodyVel_bodyFrame = zeros(length(logElements(idx_lve(1)).data), 3);
bodyAngVel_bodyFrame = zeros(length(logElements(idx_lve(1)).data), 3);

for t=1:length(logElements(idx_lve(1)).data)

    vel = [logElements(idx_lve(1)).data(t,1), logElements(idx_lve(2)).data(t,1), logElements(idx_lve(3)).data(t,1)];
    angVel = [logElements(idx_lve(1)).data(t,1), logElements(idx_lve(2)).data(t,1), logElements(idx_lve(3)).data(t,1)];
    quat = [logElements(idx_ori(1)).data(t,1), logElements(idx_ori(2)).data(t,1), logElements(idx_ori(3)).data(t,1), logElements(idx_ori(4)).data(t,1)];
    rot = quat2mat(quat);
    vel_body = rot'*vel';
    bodyVel_bodyFrame(t,:) = vel_body';
end

       idx_tor = [idx_model_meas_jointTor_LF_HAA, ...
           idx_model_meas_jointTor_LF_HFE, ...
           idx_model_meas_jointTor_LF_KFE, ...
           idx_model_meas_jointTor_LH_HAA, ...
           idx_model_meas_jointTor_LH_HFE, ...
           idx_model_meas_jointTor_LH_KFE, ...
           idx_model_meas_jointTor_RF_HAA, ...
           idx_model_meas_jointTor_RF_HFE, ...
           idx_model_meas_jointTor_RF_KFE, ...
           idx_model_meas_jointTor_RH_HAA, ...
           idx_model_meas_jointTor_RH_HFE, ...
           idx_model_meas_jointTor_RH_KFE];
       
       
readData = csvread("data/seaModelValidate_highspeed.txt");

length(readData)
Offset = 16;
Start = 1;

time2 = time(Offset:end - 1);

figure('rend','painters','pos',[10 10 600 280])
set(gca,'FontSize',10) 
set(gca,'Units','pixels') 
set(gca, 'Position', [50 50 550 230])
length(readData);
length(time2);

plot(time2(1:length(readData)), readData)

hold on  


for i=1:length(joint_idx)   
   ideal_PD = 50 * (logElements(idx_com(joint_idx(i))).data - logElements(idx_pos(joint_idx(i))).data) - 0.1 * logElements(idx_vel(joint_idx(i))).data;
    plot(time, ideal_PD)  
    plot(time, logElements(idx_tor(joint_idx(i))).data)
    
end

mse = 0;
mse2 = 0;


idx_ = joint_idx(1);

for i=1:length(readData)

    mse = mse + (readData(i) - logElements(idx_tor(idx_)).data(Offset + i - 1))^2;
   mse2 = mse2 + (ideal_PD(i) - logElements(idx_tor(idx_)).data(i))^2;

end


mse = mse / length(readData);
rmse = sqrt(mse)

mse2 = mse2 / length(readData);
rmse2 = sqrt(mse2)




axis([time_start time_end -38 22.5])
legend('Learned Model','Ideal PD Controller','Measured')

xlabel("Time [s]", 'FontSize', 10)
ylabel("Torque [Nm]", 'FontSize', 10)

% title('joint torque')

idx_vel = [idx_model_meas_jointVel_LF_HAA, ...
           idx_model_meas_jointVel_LF_HFE, ...
           idx_model_meas_jointVel_LF_KFE, ...
           idx_model_meas_jointVel_LH_HAA, ...
           idx_model_meas_jointVel_LH_HFE, ...
           idx_model_meas_jointVel_LH_KFE, ...
           idx_model_meas_jointVel_RF_HAA, ...
           idx_model_meas_jointVel_RF_HFE, ...
           idx_model_meas_jointVel_RF_KFE, ...
           idx_model_meas_jointVel_RH_HAA, ...
           idx_model_meas_jointVel_RH_HFE, ...
           idx_model_meas_jointVel_RH_KFE];

function M = quat2mat(q)

M=zeros(3,3);
M(1,1) = q(1) * q(1) + q(2) * q(2) - q(3) * q(3) - q(4) * q(4);
M(1,2) =  2 * q(2) * q(3) - 2 * q(1) * q(4);
M(1,3) = 2 * q(1) * q(3) + 2 * q(2) * q(4);

M(2,1) = 2 * q(1) * q(4) + 2 * q(2) * q(3);
M(2,2) = q(1) * q(1) - q(2) * q(2) + q(3) * q(3) - q(4) * q(4);
M(2,3) = 2 * q(3) * q(4) - 2 * q(1) * q(2);

M(3,1) = 2 * q(2) * q(4) - 2 * q(1) * q(3);
M(3,2) = 2 * q(1) * q(2) + 2 * q(3) * q(4);
M(3,3) = q(1) * q(1) - q(2) * q(2) - q(3) * q(3)+ q(4) * q(4);
 
        
end


