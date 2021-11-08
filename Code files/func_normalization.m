function [ norm_data ] = func_normalization( a,b,total_data )
% This function normalises the data between 0.1 to 0.9

% n = input('enter the total number of column to be normalised \n');
[m,n] = size(total_data);

for i = 1:m

norm_data(i,1:n)= a + (b-a)* (total_data(i,1:n) - min(total_data(i,1:n)))/(max(total_data(i,1:n)) - min(total_data(i,1:n)));

end


%[n,m] = size(total_data);
%for i = 1:n

%norm_data(1:m,i)= a + (b-a)* (total_data(1:m,i) - min(total_data(1:m,i)))/(max(total_data(1:m,i)) - min(total_data(1:m,i)));



%end

