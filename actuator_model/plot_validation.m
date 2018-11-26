close all

joint_idx = [3];
time_start = -0.05;
time_end = 5.0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
output0 = csvread("data/seaModelValidate_output0.txt"); 
output1 = csvread("data/seaModelValidate_output1.txt");

length(output0)
time = linspace(0, length(output0), length(output0));
time = time/400;
% time = time - 411.4;
Offset = 16;

figure('rend','painters','pos',[10 10 1200 280])
set(gca,'FontSize',10) 
set(gca,'Units','pixels') 
set(gca, 'Position', [50 50 1150 230])

plot(time, output0)
hold on  
plot(time, output1)

legend('Learned Model','Ideal PD Controller')

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
            
mse = 0;


idx_ = joint_idx(1);

for i=1:length(output0)

    mse = mse + (output0(i) - output1(i))^2;
end


mse = mse / length(output0);
rmse = sqrt(mse)

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


