%============================================================================ 
% This code is to generate the fixed initial seed for a particular architecture
% The code is run for 1 iterations using BP algorithm and then this
% obtained weight is used in other ANN modeling
% Every output folder will create 'seed_no of hidden neuron' mat file and will be send
% to the number of set you want as in folder 'Seed_run run no'
%============================================================================

% clc;
% clear all

% load CNRMCM5_JJAS_NORTHERN;
% load data_4_N_1
X = x'; % Raw inputs
Y = Y_allGrid(:,1)';  % Raw targets
no_Input_var = size(X,1);

% Normalizing whole data between 0.1 and 0.9 
a = 0.1;
b = 0.9;
normalized_inputs = func_normalization(a,b,X);
normalized_targets = func_normalization(a,b,Y);

inputs = normalized_inputs;
targets = normalized_targets;

% Create a Fitting Network

hiddenLayerSize = 5:5:40; % change to create the set of initial weight
% hiddenLayerSize = 3*ones(1,20);
% inputFolder = 'seed_'; % for loading the initial seed folders
outputFolder = 'seed_'; % for saving and moving the results and outputs of workspace in previously specified folder 

for i=1:numel(hiddenLayerSize)
% net = fitnet(hiddenLayerSize(i));
net = newff(inputs,targets,{hiddenLayerSize(i)},{'logsig','purelin'},'trainlm','','mse',{},{},'dividerand');
net = configure(net,inputs,targets);
% load initial_weight_4;

net.divideFcn = 'divideind';  % Divide data randomly
net.divideParam.trainInd = 1:697; % 50/100;
net.divideParam.valInd = 698:775;%4745:5645;%
net.divideParam.testInd = 776:1472; % 50/100;


%--- transfer function at hidden and output layers -------
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'purelin';

% ========================================================

net.inputs{1}.processFcns = {};
net.outputs{1}.processFcns = {};
% For help on training function 'trainlm' type: help trainlm
% For a list of all training functions type: help nntrain

net.trainFcn = 'trainlm'; %'traingdx'; %'trainlm';  % Levenberg-Marquardt
net.trainParam.goal = 0.0001;
net.trainParam.epochs = 1;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean squared error

% Train the Network
[net,tr] = train(net,inputs,targets);

IW = net.IW{1,1};
LW = net.LW{2,1};
b1 = net.b{1,1};
b2 = net.b{2,1};
% save('seed_3_run2','IW','LW','b1','b2')

outputFilename = sprintf('%s%d.mat', outputFolder, i);
save(outputFilename,'IW','LW','b1','b2')   % save data 
% movefile(outputFilename,'Seed_run8')
end 
 



