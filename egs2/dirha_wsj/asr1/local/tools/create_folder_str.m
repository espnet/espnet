%%%%%%%%%%%%%%%%%%%%%%
% Author: Mirco Ravanelli (mravanelli@fbk.eu)
%
% Description:
% This script generate a folder (folder_new) with the same strcuture of the input folder (folder_old)
%
%%%%%%%%%%%%%%%%%%%%%%
function create_folder_str(folder_old,folder_new)

paths=dir(folder_old);
paths=paths(3:end);

sum=0;

for i=1:length(paths)

    if paths(i).isdir==1

    sum=sum+1;

    folder_new_sub=strcat(folder_new,'/',paths(i).name);
    folder_old_sub=strcat(folder_old,'/',paths(i).name);
    mkdir(folder_new_sub)
    create_folder_str(folder_old_sub,folder_new_sub);

    end

end

if sum==0
return
end


end
