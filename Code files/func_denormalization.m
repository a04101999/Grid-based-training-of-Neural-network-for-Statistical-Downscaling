function [ denorm_data ] = func_denormalization( a,b,obs_out,mod_out )
% This function normalises the data between 0.1 to 0.9

% n = input('enter the total number of column to be normalised \n');
% [m,n] = size(obs_out);
[m,n] = size(mod_out);

for i = 1:m

denorm_data(i,1:n)= (mod_out(i,1:n) - a).*(max(obs_out) - min(obs_out))/(b-a) + min(obs_out);

end

