%%%%%%%%%%%%%%%%%%%%%%
% Author: Mirco Ravanelli (mravanelli@fbk.eu)
%
% Modified from the script for dirha_wsj in Espnet
% Ruizhi Li 09/07/2019
%
%%%%%%%%%%%%%%%%%%%%%%
function fileList = find_files(dirName,ext)

  dirData = dir(dirName);      %# Get the data for the current directory
  dirIndex = [dirData.isdir];  %# Find the index for directories
  fileList = {dirData(~dirIndex).name}';  %'# Get a list of the files
  if ~isempty(fileList)
    fileList = cellfun(@(x) fullfile(dirName,x),...  %# Prepend path to files
                       fileList,'UniformOutput',false);
  end
  subDirs = {dirData(dirIndex).name};  %# Get a list of the subdirectories
  validIndex = ~ismember(subDirs,{'.','..'});  %# Find index of subdirectories
                                               %#   that are not '.' or '..'
  for iDir = find(validIndex)                  %# Loop over valid subdirectories
    nextDir = fullfile(dirName,subDirs{iDir});    %# Get the subdirectory path
    fileList = [fileList; find_files(nextDir,ext)];  %# Recursively call getAllFiles
  end

   fileListOut=[];
   count=0;
   for i=1:length(fileList)
     if length(strfind(fileList{i},ext))>0 && length(strfind(fileList{i},strcat(ext, '.')))==0
     count=count+1;
     fileListOut{count}=fileList{i};
     end
   end

   fileList=fileListOut';

end
