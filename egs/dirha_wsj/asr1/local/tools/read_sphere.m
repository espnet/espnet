function [y]=read_sphere(x)
cmd=strcat('./w_decode -f -o short_01$',  x , '$tmp.wav');
cmd=strrep(cmd,'$',' ');
system(cmd);
fp=fopen('tmp.wav','r');
[y,n]=fread(fp,Inf,'short');
fclose(fp);
y=y(513:end);
