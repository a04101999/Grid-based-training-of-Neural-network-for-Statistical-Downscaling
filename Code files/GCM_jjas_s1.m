GCM_final=[];
for i=1:847
    outputs_GCM = net(xGCM'); %Change the input as GCM data 
    denormalized_output_gcm = func_denormalization( a,b,Y,outputs_GCM);% here also change accordingly
    GCM_final=[GCM_final; denormalized_output_gcm];
end