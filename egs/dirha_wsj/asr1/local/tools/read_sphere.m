%%%%%%%%%%%%%%%%%%%%%%
% Author: Mirco Ravanelli (mravanelli@fbk.eu)
%%%%%%%%%%%%%%%%%%%%%%
function [y]=read_sphere(path_reader,x)
cmd=strcat(path_reader,'$ -f wav$',  x , '$>tmp.wav');
cmd=strrep(cmd,'$',' ');
system(cmd);
y=audioread('tmp.wav');
