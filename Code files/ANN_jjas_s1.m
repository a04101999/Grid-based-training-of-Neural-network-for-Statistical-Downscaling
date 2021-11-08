%This is for jardine river of best seed for LM from the three runs 
%Solve an Input-Output Fitting problem with a Neural Network
% Script generated by NFTOOL
% Created Wed Apr 16 17:50:24 IST 2014
%
% This script assumes these variables are defined:
%
%   inputs - input data.
%   targets - target data.

clc;
clear all

ANN_output_grid=[];% to store values predicted by ANN
r_std_deviation_mean=[];% FOR CORRELATION 
GCM_final=[];
GCM_future=[];

difference=[];%To store difference between observed and those predicted by ANN

%xGCM=[];


%save('allGrid_Y_GCM.mat', 'xGCM');
%save('allGrid_Y.mat');
% Inputs and outputs in .mat file "Data_ANN".
%save('allGrid_Y.mat','xGCM');


load allGrid_Y; % Data_hybrid_B2; %less_Inp_Hybrid_B2 %Data_ANN;
% load data_4_N_1
X = x'; % Raw inputs

for ii = 1:847 %PLEASE NOTE ITERATIONS ONLY FROM GRID 1 TO 4
    y = Y_allGrid(:,ii);
Y = y'; % Raw targets
no_Input_var = size(X,1);

% Normalizing whole data between 0.1 and 0.9 
a = 0.1;
b = 0.9;
normalized_inputs = func_normalization(a,b,X);
normalized_targets = func_normalization(a,b,Y);

% The toolkit will take "inputs" and "targets" as for input and target
% these are normalized data and the output from toolkit will be also
% normalized. 

inputs = normalized_inputs; %X; %
targets = normalized_targets; %Y; % 

% save inputs   % these inputs are saved after normalization of input data
% save targets  % these outputs are saved after normalization of target data

% load inputs
% load targets

% Create a Fitting Network

hiddenLayerSize = 5:5:40;
% hiddenLayerSize = 3*ones(1,20);

% preallocating the variables which have to save

%----- with normalized ----------------------
errors = zeros(numel(hiddenLayerSize),numel(Y));
performance = zeros(numel(hiddenLayerSize),1);
output2 = zeros(numel(hiddenLayerSize),numel(Y));

%--normalized data error, TS pre allocation for train, test
AARE_n_trn = zeros(numel(hiddenLayerSize),1);
R_n_trn = zeros(numel(hiddenLayerSize),1);
E_n_trn = zeros(numel(hiddenLayerSize),1);
NRMSE_n_trn = zeros(numel(hiddenLayerSize),1);
perc_MF_n_trn = zeros(numel(hiddenLayerSize),1);
RMSE_n_trn = zeros(numel(hiddenLayerSize),1);
MSE_n_trn = zeros(numel(hiddenLayerSize),1);
AIC_n_trn = zeros(numel(hiddenLayerSize),1);
BIC_n_trn = zeros(numel(hiddenLayerSize),1);
TS_1_n_trn = zeros(numel(hiddenLayerSize),1);
TS_5_n_trn = zeros(numel(hiddenLayerSize),1);
TS_10_n_trn = zeros(numel(hiddenLayerSize),1);
TS_25_n_trn = zeros(numel(hiddenLayerSize),1);
TS_50_n_trn = zeros(numel(hiddenLayerSize),1);
TS_75_n_trn = zeros(numel(hiddenLayerSize),1);
TS_100_n_trn = zeros(numel(hiddenLayerSize),1);

AARE_n_tst = zeros(numel(hiddenLayerSize),1);
R_n_tst = zeros(numel(hiddenLayerSize),1);
E_n_tst = zeros(numel(hiddenLayerSize),1);
NRMSE_n_tst = zeros(numel(hiddenLayerSize),1);
perc_MF_n_tst = zeros(numel(hiddenLayerSize),1);
RMSE_n_tst = zeros(numel(hiddenLayerSize),1);
MSE_n_tst = zeros(numel(hiddenLayerSize),1);
AIC_n_tst = zeros(numel(hiddenLayerSize),1);
BIC_n_tst = zeros(numel(hiddenLayerSize),1);
TS_1_n_tst = zeros(numel(hiddenLayerSize),1);
TS_5_n_tst = zeros(numel(hiddenLayerSize),1);
TS_10_n_tst = zeros(numel(hiddenLayerSize),1);
TS_25_n_tst = zeros(numel(hiddenLayerSize),1);
TS_50_n_tst = zeros(numel(hiddenLayerSize),1);
TS_75_n_tst = zeros(numel(hiddenLayerSize),1);
TS_100_n_tst = zeros(numel(hiddenLayerSize),1);
% % All errors and TS for normalized data is saved in "ERROR_TS_Norm" of
% % 16 coulmn for training data and other 16 column is for testing data. 
ERROR_TS_Norm = zeros(numel(hiddenLayerSize),32);

%=========================================================================
%--Denormalized data error, TS pre allocation for train, test
AARE_dn_trn = zeros(numel(hiddenLayerSize),1);
R_dn_trn = zeros(numel(hiddenLayerSize),1);
E_dn_trn = zeros(numel(hiddenLayerSize),1);
NRMSE_dn_trn = zeros(numel(hiddenLayerSize),1);
perc_MF_dn_trn = zeros(numel(hiddenLayerSize),1);
RMSE_dn_trn = zeros(numel(hiddenLayerSize),1);
MSE_dn_trn = zeros(numel(hiddenLayerSize),1);
AIC_dn_trn = zeros(numel(hiddenLayerSize),1);
BIC_dn_trn = zeros(numel(hiddenLayerSize),1);
TS_1_dn_trn = zeros(numel(hiddenLayerSize),1);
TS_5_dn_trn = zeros(numel(hiddenLayerSize),1);
TS_10_dn_trn = zeros(numel(hiddenLayerSize),1);
TS_25_dn_trn = zeros(numel(hiddenLayerSize),1);
TS_50_dn_trn = zeros(numel(hiddenLayerSize),1);
TS_75_dn_trn = zeros(numel(hiddenLayerSize),1);
TS_100_dn_trn = zeros(numel(hiddenLayerSize),1);

AARE_dn_tst = zeros(numel(hiddenLayerSize),1);
R_dn_tst = zeros(numel(hiddenLayerSize),1);
E_dn_tst = zeros(numel(hiddenLayerSize),1);
NRMSE_dn_tst = zeros(numel(hiddenLayerSize),1);
perc_MF_dn_tst = zeros(numel(hiddenLayerSize),1);
RMSE_dn_tst = zeros(numel(hiddenLayerSize),1);
MSE_dn_tst = zeros(numel(hiddenLayerSize),1);
AIC_dn_tst = zeros(numel(hiddenLayerSize),1);
BIC_dn_tst = zeros(numel(hiddenLayerSize),1);
TS_1_dn_tst = zeros(numel(hiddenLayerSize),1);
TS_5_dn_tst = zeros(numel(hiddenLayerSize),1);
TS_10_dn_tst = zeros(numel(hiddenLayerSize),1);
TS_25_dn_tst = zeros(numel(hiddenLayerSize),1);
TS_50_dn_tst = zeros(numel(hiddenLayerSize),1);
TS_75_dn_tst = zeros(numel(hiddenLayerSize),1);
TS_100_dn_tst = zeros(numel(hiddenLayerSize),1);

% All errors and TS for denormalized data is saved in "ERROR_TS_Denorm"
% 16 coulmns for training data and rest 16 columns (17-32) are for testing data 

ERROR_TS_Denorm = zeros(numel(hiddenLayerSize),32);

%==========================================================================
%grid='grid_';
inputFolder = 'seed_'; % for loading the initial seed folders
outputFolder = 'results_JJAS_Northern_'; % for saving and moving the results and outputs of workspace in previously specified folder 
outputFolder1 = 'results_Grid_'; % for saving and moving the results and outputs of workspace in previously specified folder 
%open 
for i=1:numel(hiddenLayerSize)

% Create a Fitting Network
% hiddenLayerSize = 10;
% load seed_1;

% seed_i = strcat('seed_',num2str(i),'.mat'); % this command will change the name of seed_i.mat file as i=1,2,3..... 
% load(seed_i);   % this command will load seed_i.mat file as i=1,2,3..... 
%cd foldername;
inputFilename = sprintf('%s%d.mat',inputFolder,i);
load(inputFilename)

% tic
% net = fitnet(hiddenLayerSize(i));

% net = configure(net,inputs,targets);


net = newff(inputs,targets,hiddenLayerSize(i),{'logsig','logsig'},'trainlm','','mse',{},{},'dividerand');
net = configure(net,inputs,targets);
% load initial weight and bais seeds 

% IW = [0.457415121718547,0.0817771746406773,0.315788928652749,1.87085198669634;-0.929943875291595,0.670164033623218,-1.32954537067693,-0.877312564315200;-0.462219391538396,1.53472228868726,-0.363271947867992,0.836813984154120;-0.446266506205748,1.13610680903541,-0.927701285127475,-1.24574940734010];
% LW = [0.862191653747635,-0.174319747521592,0.723127408603735,-0.633189000572160];
% b1 = [-2.02702017673482;0.687486571018787;0.647341929450844;-2.12396654431958];
% b2 = 0.192311106132482;

net.IW{1,1}= IW;
net.LW{2,1} = LW;
net.b{1,1} = b1;
net.b{2,1} = b2;

% Setup Division of Data for Training, Validation, Testing
% net.divideParam.trainRatio = 70/100;
% net.divideParam.valRatio = 15/100;
% net.divideParam.testRatio = 15/100;

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'divideind';  % Divide data randomly
net.divideParam.trainInd = 1:697; % 50/100;
net.divideParam.valInd = 698:775;%4745:5645;%
net.divideParam.testInd = 776:1472; % 50/100;

%--- transfer function at hidden and output layers -------
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig'; %purelin

% ========================================================

net.inputs{1}.processFcns = {};
net.outputs{1}.processFcns = {};
% ========================================================

net.trainFcn= 'trainlm'; %'traingda'; %'traingdx'; %'traingdm';
net.trainParam.goal = 0.0001;
net.trainParam.epochs = 5000;
net.trainParam.min_grad = 0;
% net.trainParam.lr = 0.005;
% net.trainParam.mc = 0.075;
% net.trainParam.mu  = 0.001; %0.001 default 
% Train the Network

[net,tr] = train(net,inputs,targets);
% toc
% Test the Network
IW1 = net.IW{1,1};
LW1 = net.LW{2,1};
b11 = net.b{1,1};
b21 = net.b{2,1};
% Input_at = inputs;
outputs = net(inputs);
errors(i,:) = gsubtract(targets,outputs);
performance(i,:) = perform(net,targets,outputs);
output2(i,:) = net(inputs);
%=======================
%  net_inp_hdn = (IW1*(inputs))+ b11
% net_inp_hdn = LW1\(outputs-b21);
%%=======================================================================================
        % Error and TS statistics usinng normalized modeled outputs for training
%----------------------------------------------------------------------------------------
[AARE_n_trn(i,1),R_n_trn(i,1),E_n_trn(i,1),NRMSE_n_trn(i,1),perc_MF_n_trn(i,1),RMSE_n_trn(i,1),MSE_n_trn(i,1)] = func_perfm_para(targets(1,tr.trainInd) ,output2(i,tr.trainInd) ,numel(targets(1,tr.trainInd)));
AIC_n_trn(i,1) = numel(targets(1,tr.trainInd))*log(RMSE_n_trn(i,1))+2*((i)*(no_Input_var+2)+1);
BIC_n_trn(i,1) = numel(targets(1,tr.trainInd))*log(RMSE_n_trn(i,1))+((i)*(no_Input_var+2)+1)*log(numel(targets(1,tr.trainInd)));

[TS_1_n_trn(i,1), TS_5_n_trn(i,1),TS_10_n_trn(i,1), TS_25_n_trn(i,1),TS_50_n_trn(i,1),TS_75_n_trn(i,1),TS_100_n_trn(i,1)] = func_TS( targets(1,tr.trainInd) ,output2(i,tr.trainInd) ,numel(targets(1,tr.trainInd)));
%-----------------------------------------------------------------------------------------
    % Error and TS statistics using normalized modeled outputs for testing data
%-----------------------------------------------------------------------------------------
[AARE_n_tst(i,1),R_n_tst(i,1),E_n_tst(i,1),NRMSE_n_tst(i,1),perc_MF_n_tst(i,1),RMSE_n_tst(i,1),MSE_n_tst(i,1)] = func_perfm_para(targets(1,tr.testInd) ,output2(i,tr.testInd) ,numel(targets(1,tr.testInd)));
AIC_n_tst(i,1) = numel(targets(1,tr.testInd))*log(RMSE_n_tst(i,1))+2*((i)*(no_Input_var+2)+1);
BIC_n_tst(i,1) = numel(targets(1,tr.testInd))*log(RMSE_n_tst(i,1))+((i)*(no_Input_var+2)+1)*log(numel(targets(1,tr.testInd)));

[TS_1_n_tst(i,1), TS_5_n_tst(i,1),TS_10_n_tst(i,1), TS_25_n_tst(i,1),TS_50_n_tst(i,1),TS_75_n_tst(i,1),TS_100_n_tst(i,1)]  = func_TS( targets(1,tr.testInd) ,output2(i,tr.testInd) ,numel(targets(1,tr.testInd)));

%-------------------------------------------------------------------------
              % Saving errors of train normalized data
%-------------------------------------------------------------------------
ERROR_TS_Norm(i,1) = AARE_n_trn(i,1);
ERROR_TS_Norm(i,2) = R_n_trn(i,1);
ERROR_TS_Norm(i,3) = E_n_trn(i,1);
ERROR_TS_Norm(i,4) = NRMSE_n_trn(i,1);
ERROR_TS_Norm(i,5) = RMSE_n_trn(i,1); 
ERROR_TS_Norm(i,6) = MSE_n_trn(i,1);
ERROR_TS_Norm(i,7) = perc_MF_n_trn(i,1);
ERROR_TS_Norm(i,8) = AIC_n_trn(i,1);
ERROR_TS_Norm(i,9) = BIC_n_trn(i,1);
ERROR_TS_Norm(i,10) = TS_1_n_trn(i,1);
ERROR_TS_Norm(i,11) = TS_5_n_trn(i,1);
ERROR_TS_Norm(i,12) = TS_10_n_trn(i,1);
ERROR_TS_Norm(i,13) = TS_25_n_trn(i,1);
ERROR_TS_Norm(i,14) = TS_50_n_trn(i,1);
ERROR_TS_Norm(i,15) = TS_75_n_trn(i,1);
ERROR_TS_Norm(i,16) = TS_100_n_trn(i,1);
%-------------------------------------------------------------------------
          % Saving errors of test normalized data
%-------------------------------------------------------------------------
ERROR_TS_Norm(i,17) = AARE_n_tst(i,1);
ERROR_TS_Norm(i,18) = R_n_tst(i,1);
ERROR_TS_Norm(i,19) = E_n_tst(i,1);
ERROR_TS_Norm(i,20) = NRMSE_n_tst(i,1);
ERROR_TS_Norm(i,21) = RMSE_n_tst(i,1); 
ERROR_TS_Norm(i,22) = MSE_n_tst(i,1);
ERROR_TS_Norm(i,23) = perc_MF_n_tst(i,1);
ERROR_TS_Norm(i,24) = AIC_n_tst(i,1);
ERROR_TS_Norm(i,25) = BIC_n_tst(i,1);
ERROR_TS_Norm(i,26) = TS_1_n_tst(i,1);
ERROR_TS_Norm(i,27) = TS_5_n_tst(i,1);
ERROR_TS_Norm(i,28) = TS_10_n_tst(i,1);
ERROR_TS_Norm(i,29) = TS_25_n_tst(i,1);
ERROR_TS_Norm(i,30) = TS_50_n_tst(i,1);
ERROR_TS_Norm(i,31) = TS_75_n_tst(i,1);
ERROR_TS_Norm(i,32) = TS_100_n_tst(i,1);
%%===========================================================================
% %----------------Denormalization of modeled outputs -----------------------
% denormalized_output = func_denormalization( a,b,Y,outputs(1,1:4744));
  denormalized_output(i,:) = func_denormalization( a,b,Y,output2(i,:));
%%===========================================================================
  % Error and TS statistics usinng denormalized modeled outputs for training
%-------------------------------------------------------------------------------
[AARE_dn_trn(i,1),R_dn_trn(i,1),E_dn_trn(i,1),NRMSE_dn_trn(i,1),perc_MF_dn_trn(i,1),RMSE_dn_trn(i,1),MSE_dn_trn(i,1)] = func_perfm_para(Y(1,tr.trainInd) ,denormalized_output(i,tr.trainInd) ,numel(Y(1,tr.trainInd)));
AIC_dn_trn(i,1) = numel(Y(1,tr.trainInd))*log(RMSE_dn_trn(i,1))+2*((i)*(no_Input_var+2)+1);
BIC_dn_trn(i,1) = numel(Y(1,tr.trainInd))*log(RMSE_dn_trn(i,1))+((i)*(no_Input_var+2)+1)*log(numel(Y(1,tr.trainInd)));

[TS_1_dn_trn(i,1), TS_5_dn_trn(i,1),TS_10_dn_trn(i,1), TS_25_dn_trn(i,1),TS_50_dn_trn(i,1),TS_75_dn_trn(i,1),TS_100_dn_trn(i,1)] = func_TS( Y(1,tr.trainInd) ,denormalized_output(i,tr.trainInd) ,numel(Y(1,tr.trainInd)));
%--------------------------------------------------------------------------------
  % Error and TS statistics by using denormalized modeled outputs for testing data
%--------------------------------------------------------------------------------
[AARE_dn_tst(i,1),R_dn_tst(i,1),E_dn_tst(i,1),NRMSE_dn_tst(i,1),perc_MF_dn_tst(i,1),RMSE_dn_tst(i,1),MSE_dn_tst(i,1)] = func_perfm_para(Y(1,tr.testInd) ,denormalized_output(i,tr.testInd) ,numel(Y(1,tr.testInd)));
AIC_dn_tst(i,1) = numel(Y(1,tr.testInd))*log(RMSE_dn_tst(i,1))+2*((i)*(no_Input_var+2)+1);
BIC_dn_tst(i,1) = numel(Y(1,tr.testInd))*log(RMSE_dn_tst(i,1))+((i)*(no_Input_var+2)+1)*log(numel(Y(1,tr.testInd)));

[TS_1_dn_tst(i,1), TS_5_dn_tst(i,1),TS_10_dn_tst(i,1), TS_25_dn_tst(i,1),TS_50_dn_tst(i,1),TS_75_dn_tst(i,1),TS_100_dn_tst(i,1)]  = func_TS( Y(1,tr.testInd) ,denormalized_output(i,tr.testInd) ,numel(Y(1,tr.testInd)));

%-------------------------------------------------------------------------
               % Saving errors of train denormalized data
%-------------------------------------------------------------------------
ERROR_TS_Denorm(i,1) = AARE_dn_trn(i,1);
ERROR_TS_Denorm(i,2) = R_dn_trn(i,1);
ERROR_TS_Denorm(i,3) = E_dn_trn(i,1);
ERROR_TS_Denorm(i,4) = NRMSE_dn_trn(i,1);
ERROR_TS_Denorm(i,5) = RMSE_dn_trn(i,1); 
ERROR_TS_Denorm(i,6) = MSE_dn_trn(i,1);
ERROR_TS_Denorm(i,7) = perc_MF_dn_trn(i,1);
ERROR_TS_Denorm(i,8) = AIC_dn_trn(i,1);
ERROR_TS_Denorm(i,9) = BIC_dn_trn(i,1);
ERROR_TS_Denorm(i,10) = TS_1_dn_trn(i,1);
ERROR_TS_Denorm(i,11) = TS_5_dn_trn(i,1);
ERROR_TS_Denorm(i,12) = TS_10_dn_trn(i,1);
ERROR_TS_Denorm(i,13) = TS_25_dn_trn(i,1);
ERROR_TS_Denorm(i,14) = TS_50_dn_trn(i,1);
ERROR_TS_Denorm(i,15) = TS_75_dn_trn(i,1);
ERROR_TS_Denorm(i,16) = TS_100_dn_trn(i,1);
%-------------------------------------------------------------------------
               % Saving errors of test denormalized data
%-------------------------------------------------------------------------
ERROR_TS_Denorm(i,17) = AARE_dn_tst(i,1);
ERROR_TS_Denorm(i,18) = R_dn_tst(i,1);
ERROR_TS_Denorm(i,19) = E_dn_tst(i,1);
ERROR_TS_Denorm(i,20) = NRMSE_dn_tst(i,1);
ERROR_TS_Denorm(i,21) = RMSE_dn_tst(i,1); 
ERROR_TS_Denorm(i,22) = MSE_dn_tst(i,1);
ERROR_TS_Denorm(i,23) = perc_MF_dn_tst(i,1);
ERROR_TS_Denorm(i,24) = AIC_dn_tst(i,1);
ERROR_TS_Denorm(i,25) = BIC_dn_tst(i,1);
ERROR_TS_Denorm(i,26) = TS_1_dn_tst(i,1);
ERROR_TS_Denorm(i,27) = TS_5_dn_tst(i,1);
ERROR_TS_Denorm(i,28) = TS_10_dn_tst(i,1);
ERROR_TS_Denorm(i,29) = TS_25_dn_tst(i,1);
ERROR_TS_Denorm(i,30) = TS_50_dn_tst(i,1);
ERROR_TS_Denorm(i,31) = TS_75_dn_tst(i,1);
ERROR_TS_Denorm(i,32) = TS_100_dn_tst(i,1);

% %%===========================END=====
===========================================
% 
% %------another method for error calculations ------------------------------
% 
% % dividing the training and testing target and outputs
% % 
% % train_target = Y(:,tr.trainInd);
% % test_target = Y(:,tr.testInd);
% % 
% % denorm_train_output(i,) = denormalized_output(:,tr.trainInd);
% % denorm_test_output = denormalized_output(:,tr.testInd);
% 
% % AARE=sum(abs((target-mod_out)./target))*100/no_of_observed_data;
% 
% % AARE(i,1) = sum(abs((Y(1,tr.trainInd)-denormalized_output(i,tr.trainInd))./Y(1,tr.trainInd)))*100/numel(Y(1,tr.trainInd));
% % 
% % R(i,1) = sum((denormalized_output(i,tr.trainInd) - mean(denormalized_output(i,tr.trainInd))).*(Y(1,tr.trainInd)...
% %  - mean(Y(1,tr.trainInd))))/sqrt(sum((denormalized_output(i,tr.trainInd) -...
% %  mean(denormalized_output(i,tr.trainInd))).^2).*sum((Y(1,tr.trainInd) - mean(Y(1,tr.trainInd))).^2));
% % 
% % NRMSE(i,1) = (sum((denormalized_output(i,tr.trainInd)-Y(1,tr.trainInd)).^2)/numel(Y(1,tr.trainInd))^0.5/(mean(Y(1,tr.trainInd))));
% % 
% % RMSE (i,1)=(sum((Y(1,tr.trainInd) - denormalized_output(i,tr.trainInd)).^2 )/numel(Y(1,tr.trainInd)))^0.5;
% % 
% % E_2(i,1) = sum((Y(1,tr.trainInd)-denormalized_output(i,tr.trainInd)).^2);
% % E_1(i,1)= sum((Y(1,tr.trainInd)-mean(Y(1,tr.trainInd))).^2);
% % E(i,1) = 1-(E_2(i,1)/E_1(i,1));
% % 
% % ARE(i,:) = (abs((Y(1,tr.trainInd) - denormalized_output(i,tr.trainInd))./Y(1,tr.trainInd))*100);
% 
% % [TS_1, TS_5,TS_10, TS_25,TS_50,TS_75,TS_100] = func_TS( Y(1,tr.trainInd) ,denormalized_output(i,tr.trainInd) ,numel(Y(1,tr.trainInd)));
% 
% % View the Network
% % view(net)
% % save traingdx_1  
% % movefile('traingdx_1.mat','Train_gdx')
% 
%---------------------------------------
% mse_trn =(sum(( trn_data_out - trn_NN_out).^2))/trn_pattern;
%     mse_fnl_trn(itr,1) = mse_trn;
%     if mse_fnl_trn (itr,1) < 0.0001
%         fprintf( 'Terminate due to destination error reached \n')
%         break
%     elseif itr==50001;
%         fprintf( 'Termination \n')
%         break
%     end
%     
%     fprintf('completed iterations %f\n',itr)

if (R_dn_trn(i,1)>= R_dn_tst(i,1)) && (R_dn_tst(i,1)>= 0.35)
    break
end
% if(R_dn_trn(i,1)>= R_dn_tst(i,1)) && (R_dn_tst(i,1)>= 0.4)   
%     break
% end



%---------------------------------------

%# Savedata
 
 
end
outputFilename = sprintf('%s%d.mat', outputFolder, i);
save(outputFilename)  % File name changes every time for different hidden neuron
%movefile(outputFilename,'Seed_run1');

Output_Grid = denormalized_output(i,:);

%disp(Output_Grid);

_grid = [ANN_output_grid;Output_Grid];%for output from ann
%disp(R_dn_trn(i,1));

outputs_GCM = net(xGCM'); %Change the input as GCM data 
denormalized_output_gcm = func_denormalization( a,b,Y,outputs_GCM);% here also change accordingly
GCM_final= [GCM_final;denormalized_output_gcm];

outputs_GCMfuture = net(xGCMfuture'); %Change the input as GCM data 
denormalized_output_gcmfuture = func_denormalization( a,b,Y,outputs_GCMfuture);% her
GCM_future= [GCM_future;denormalized_output_gcmfuture];

r_std_deviation_mean(ii,1)=R_dn_trn(i,1);
r_std_deviation_mean(ii,2)=R_dn_tst(i,1);
r_std_deviation_mean(ii,4)= std(Output_Grid(1:775));
r_std_deviation_mean(ii,5)=std(Output_Grid(776:1472));
r_std_deviation_mean(ii,7)= mean(Output_Grid(1:775));
r_std_deviation_mean(ii,8)=mean(Output_Grid(776:1472));
obs_mean_trn= mean(y(1:775));
obs_mean_tst=mean(y(776:1472));
mean_trn=mean(Output_Grid(1:775));
mean_tst=mean(Output_Grid(776:1472));
difference(ii,1)=abs(obs_mean_trn-mean_trn);
difference(ii,2)=abs(obs_mean_tst-mean_tst);

outputFilename1 = sprintf('%s%d.mat', outputFolder1, ii);
save(outputFilename1); 
end
save('statistics_for_idx_1.mat', 'r_std_deviation_mean','difference');
save('ANN_output_rainfall_for_idx_1.mat', 'ANN_output_grid');

% % ======================================= END ====================================================
% 
% % save('results_ANN_GDM_LMPD_Kenntucky_ALL')

% % TS = zeros(numel(hiddenLayerSize),7);
% % 
% % for i=1:numel(hiddenLayerSize)
% % [TS_1(i,1), TS_5(i,1),TS_10(i,1), TS_25(i,1),TS_50(i,1),TS_75(i,1),TS_100(i,1)] = func_TS( Y(1,tr.trainInd) ,denormalized_output(i,tr.trainInd) ,numel(Y(1,tr.trainInd)));
% % % TS_trn = {'TS_1', 'TS_5','TS_10', 'TS_25','TS_50','TS_75','TS_100';TS_1 TS_5 TS_10 TS_25 TS_50 TS_75 TS_100} ;
% % 
% % [AARE(i,1),R(i,1),E(i,1),NRMSE(i,1),perc_MF(i,1),RMSE(i,1),MSE(i,1)] = func_perfm_para(Y(1,tr.trainInd) ,denormalized_output(i,tr.trainInd) ,numel(Y(1,tr.trainInd)));
% % 
% % 
% % 
% % TS(i,1) = TS_1(i,1);
% % TS(i,2) = TS_5(i,1);
% % TS(i,3) = TS_10(i,1);
% % TS(i,4) = TS_25(i,1);
% % TS(i,5) = TS_50(i,1);
% % TS(i,6) = TS_75(i,1);
% % TS(i,7) = TS_100(i,1);
% % end
% 
% % for j = 1:numel(hiddenLayerSize)
% % c1 = 0;
% % % c2 = 0;
% % for k = 1:numel(Y(1,tr.trainInd))
% %     if  ARE(j,k) < 1
% %     c1 = c1+1;
% % %     c2 = c2+1;
% %     end 
% % end
% % TS_1(j,1) = c1*100/numel(Y(1,tr.trainInd));
% % end 
% 
% 
%     
%     
%     
%     
% % %-------- denoramized outputs,training and testing outputs
% % 
% % % denormalized_output = func_denormalization( a,b,Y,output2 ); % Y is denormalized target and output2 is calculated normalized output 
% % 
% % %% =================================================================================
% % % extracting the data of training, validation and testing from denormalized targets and outputs
% % % and normalized error by using  tr.trainInd, tr.valInd and tr.testInd contain the indices of the 
% % % data points that were used in the training, validation and test sets, respectively.
% % %% =================================================================================
% % 
% % train_target = Y(:,tr.trainInd);
% % test_target = Y(:,tr.testInd);
% % 
% % denorm_train_output = denormalized_output(:,tr.trainInd);
% % denorm_test_output = denormalized_output(:,tr.testInd);
% % 
% % 
% % normalized_error_train = errors(:,tr.trainInd);
% % normalized_error_test = errors(:,tr.testInd);
% % 
% % %% ====================Normalized data===============================================
% % %  Performance statistics by normalized error data for training 
% % %
% % %% ==================================================================================
% % 
% %   
% % 
% % 
% % %% ==================================================================================
% % %  Performance statistics by normalized error data for testing 
% % %
% % %% ==================================================================================
% % 
% % 
% % 
% % 
% % 
% % %% ====================Denormalized data===============================================
% % %  Performance statistics by denormalized data for training 
% % %
% % %% ==================================================================================
% % 
% % %% %%% training statistics on denormalized data  %%%%%%%%
% % % fprintf('     Model Performances parameters During TRAINING on denormalized data\n  ')
% % [k,l] = size(train_target); 
% % for i = 1:b
% %   [AARE(i),R(i),E(i),NRMSE(i),perc_MF(i),RMSE(i),MSE(i)]= func_perfm_para( train_target,denorm_train_output(i,:),l );
% %   trn_perform_denorm(i,:) ={ AARE(i),R(i),E(i),NRMSE(i),perc_MF(i),RMSE(i),MSE(i)};
% % end
% %   
% % 
% % 
% % %% ==================================================================================
% % %  Performance statistics by denormalized data for testing 
% % %
% % %% ==================================================================================
% % 
% % 
% % 
% % 
% % 
% % % Plots
% % % Uncomment these lines to enable various plots.
% % %figure, plotperform(tr)
% % %figure, plottrainstate(tr)
% % %figure, plotfit(net,inputs,targets)
% % %figure, plotregression(targets,outputs)
% % %figure, ploterrhist(errors)
%Major modification s for the basic selfi multiplication factore