%%%%%%%%%%%%%%%%%%%%%%
% Author: Mirco Ravanelli (mravanelli@fbk.eu)
%%%%%%%%%%%%%%%%%%%%%%
function [ x_shift ] = linear_shift(x,p)


if p>=0
x_shift=circshift(x',[p,0])';
x_shift(1:p)=0;
else
x_shift=circshift(x',[p,0])';
x_shift(end+p+1:end)=0;
end



end
