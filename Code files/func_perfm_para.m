%% This is function for calculating the PERFORMANCE of MODELS
function [AARE,R,E,NRMSE,perc_MF,RMSE,MSE] = func_perfm_para( target ,mod_out,no_of_observed_data )

              %%%%%%%%%%%%%%%%%%%%  AARE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
AARE=sum(abs((target-mod_out)./target))*100/no_of_observed_data;
% fprintf('AARE is %f\n',AARE)

              %%%%%%%%%%%%%%%%%%%%  R  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
R = sum((mod_out - mean(mod_out)).*(target - mean(target)))/sqrt(sum((mod_out - mean(mod_out)).^2).*sum((target - mean(target)).^2));
% R = corr(target,mod_out);
% fprintf('Correlation coefficient is %f\n',R)

              %%%%%%%%%%%%%%%%%%%%  E  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 E_2 = sum((target-mod_out).^2);
 E_1 = sum((target-mean(target)).^2);
 E = 1-(E_2/E_1);
% fprintf('NASH SUTCLIFF EFFICIENCY (E) is %f\n',E)

              %%%%%%%%%%%%%%%%%%%%  NRMSE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NRMSE = (sum((mod_out-target).^2)/no_of_observed_data)^0.5/(mean(target));
% fprintf('NRMSE is %f\n',NRMSE)

              %%%%%%%%%%%%%%%%%%%%   %MF   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
perc_MF = (max(mod_out) - max(target))*100/max(target);
% fprintf('perc_MF is %f\n',perc_MF)

              %%%%%%%%%%%%%%%%%%%%   RMSE   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RMSE=(sum((target - mod_out).^2 )/no_of_observed_data)^0.5;
% fprintf('RMSE is %f\n',RMSE)

              %%%%%%%%%%%%%%%%%%%%   MSE   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 MSE = sum(( target - mod_out).^2)/no_of_observed_data;
% fprintf('MSE is %f\n',MSE)
end

