function [TS_1, TS_5,TS_10, TS_25,TS_50,TS_75,TS_100] = func_TS( target ,mod_out ,no_of_observed_data)

ARE = (abs((target - mod_out)./target)*100);
%%
c = 0;
for i = 1:no_of_observed_data
if  ARE(1,i) < 1
    c = c+1;
end 
end
TS_1 = c*100/no_of_observed_data;
% fprintf ('TS_1 is %f\n',TS_1)
%%
b = 0;
for i = 1:no_of_observed_data
if ARE(1,i) < 5
    b = b+1;
end 
end
TS_5 = b*100/no_of_observed_data;
% fprintf ('TS_5 is %f\n',TS_5)
%%
c = 0;
for i = 1:no_of_observed_data
if ARE(1,i) < 10
    c = c+1;
end 
end
TS_10 = c*100/no_of_observed_data;
% fprintf ('TS_10 is %f\n',TS_10)
%%
d = 0;
for i = 1:no_of_observed_data
if ARE(1,i) < 25
    d=d+1;
end 
end
TS_25 = d*100/no_of_observed_data;
% fprintf ('TS_25 is %f\n',TS_25)
%%
e = 0;
for i = 1:no_of_observed_data
if ARE(1,i) < 50
    e = e+1;
end 
end
TS_50 = e*100/no_of_observed_data;
% fprintf ('TS_50 is %f\n',TS_50)
%%
f  =0;
for i=1:no_of_observed_data
if ARE(1,i) < 75
   f = f+1;
end 
end
TS_75=f*100/no_of_observed_data;
% fprintf ('TS_75 is %f\n',TS_75)

%%
g = 0;
for i = 1:no_of_observed_data
if ARE(1,i) < 100
    g=g+1;
end 
end
TS_100 = g*100/no_of_observed_data;
% fprintf ('TS_100 is %f\n',TS_100)
end
