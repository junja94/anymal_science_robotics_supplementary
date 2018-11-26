close all 
joint_idx = [6];

idx_tor = [
           idx_model_meas_jointTor_LF_HAA, ...
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

idx_pos = [
    idx_model_meas_jointPos_LF_HAA, ...
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

idx_vel = [
    idx_model_meas_jointVel_LF_HAA, ...
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

idx_com = [ 
    idx_command_desJointPos_LF_HAA, ...
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

        
%      for i=1:12
%       idx_position = idx_pos(i);
%       logElements(idx_position).data(1)
%      end
%     logElements(idx_model_meas_orientBaseToWorldQuat_w).data(1) 
%     logElements(idx_model_meas_orientBaseToWorldQuat_x).data(1)
%     logElements(idx_model_meas_orientBaseToWorldQuat_y).data(1)
%     logElements(idx_model_meas_orientBaseToWorldQuat_z).data(1)
%         
        
size = 0;
number_of_timeSteps = 16;
stride = 1;
data_size = 0;

for i=1:length(joint_idx)
    
    idx_torque = idx_tor(joint_idx(i));
    temp = length(logElements(idx_torque).data) - 1;
    size = size + temp;   
    data_size = data_size + temp -(number_of_timeSteps-1) * stride;
     
end  

size
data_size


torque = zeros(1, size);
position = zeros(1, size);
velocity = zeros(1, size);
command = zeros(1, size);
command2 = zeros(1, size);

data_input = zeros(3*number_of_timeSteps, data_size);
data_target = zeros(1, data_size);

idx = 1;
counter = 1;
% hold on
% plot(command)
position_error_stack = [];
velocity_stack = [];
torque_stack = [];

for i=1:length(joint_idx)
    idx_torque = idx_tor(joint_idx(i));
    idx_position = idx_pos(joint_idx(i));
    idx_velocity = idx_vel(joint_idx(i));
    idx_command = idx_com(joint_idx(i));    
    
    size = length(logElements(idx_torque).data) - stride;    
    torque = zeros(1, size);
    position = zeros(1, size);
    velocity = zeros(1, size);
    command = zeros(1, size);
    positionChange = zeros(1, size);
    
    torque(1, 1:size) = logElements(idx_torque).data(1:size)';    
    position(1, 1:size) = logElements(idx_position).data(1:size)';
    velocity(1, 1:size) = logElements(idx_velocity).data(1:size)';    
    command(1, 1:size) = logElements(idx_command).data(1:size)';
    positionChange(1, 1:size) = logElements(idx_position).data(1 + stride:size+stride)' - position;
%     positionChange = positionChange - position;
    
    positionError = command-position;
    position_error_stack = [position_error_stack , positionError];
    torque_stack = [torque_stack, torque];
    velocity_stack = [velocity_stack, velocity];
    for j=1:size-(number_of_timeSteps-1) * stride
        data_input(1:number_of_timeSteps, counter) = velocity(1, j:stride:j+(number_of_timeSteps-1) * stride)';
        data_input(number_of_timeSteps+1:2*number_of_timeSteps, counter) = positionError(1, j:stride:j+(number_of_timeSteps-1) * stride);
         data_input(2* number_of_timeSteps+1:3*number_of_timeSteps, counter) = torque(1, j:stride:j+(number_of_timeSteps-1) * stride);
    
          data_target(1,counter) = torque(1, j+(number_of_timeSteps-1) * stride);  
%         data_target(2,counter) = positionChange(1, j+(number_of_timeSteps-1) * stride);  
        counter = counter + 1;
    end   
end

time = double(logElements(1).data - logElements(1).data(1));
time = time + double(logElements(2).data) * 10^-9;


fileID = fopen(sprintf('data_input0%d.bin',fNumber), 'w');
fwrite(fileID, data_input,'double');
fclose(fileID);

fileID2 = fopen(sprintf('data_target0%d.bin',fNumber), 'w');
fwrite(fileID2, data_target, 'double');
fclose(fileID2);

% figure
data_size = length(data_target);
% figure 
% hold on
plot(data_target(1,:))
% plot(data_input(16,:))
% plot(position_error_stack);
% plot(data_target(2,:))
