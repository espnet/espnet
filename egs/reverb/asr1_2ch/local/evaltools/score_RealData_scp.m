%% score_RealData.m
%%
%% Evaluate speech enhancement results for RealData. This is
%% designed for use in the REVERB challenge. 
%%
%% Written and distributed by the REVERB challenge organizers on 1 July, 2013
%% Inquiries to the challenge organizers (REVERB-challenge@lab.ntt.co.jp)

function score_RealData_scp(senhroot,scp_file)

% Evaluation
%----------------------------------------------------------------------

addpath ./prog;
srmrdir = 'SRMRToolbox';
addpath(genpath('SRMRToolbox/libs'));

dists = {'far', 'near'};

taskdir  = '../taskfiles/1ch';
resdir   = '../scores/RealData';

for i1 = 1 : length(dists)
  dist = dists{i1};
  
  tgtlist = scp_file;
  
  %% Evaluate the quality of original data
  %name = ['dt_', dist, '_room1_orig'];    
  %score_real_scp(name, resdir, tgtlist, origroot, srmrdir);
  
  %% Evaluate the quality of enhanced data
  name = ['et_', dist, '_room1_senh'];    
  score_real_scp(name, resdir, tgtlist, senhroot, srmrdir);
end


% Creating summary
%----------------------------------------------------------------------

types   = {'senh'};
workdir = fullfile(resdir, 'work');

srmr_mean = zeros(length(dists), length(types));

for i1 = 1 : length(dists)
  for i2 = 1 : length(types)
    dist   = dists{i1};
    typ     = types{i2};

    name    = ['et_', dist, '_room1_', typ];
    resfile = fullfile(workdir, name);    
    fid     = fopen(resfile);
    while ~feof(fid)
      l = fgetl(fid);
      
      %% SRMR
      if strfind(l, 'AVG SRMR')
	[nul, l]   = strtok(strtrim(l));
	[nul, l]   = strtok(strtrim(l));
	l          = strtrim(l);
	if strcmp(l(1), ':')
	  l = strtrim(l(2 : end));
	end
	
	srmr_mean(i1, i2) = str2num(strtrim(strtok(l)));
      end      
    end
    fclose(fid);
  end
end

fid  = fopen(fullfile(fileparts(resdir), 'score_RealData'), 'w');
fids = [1, fid];

%% SRMR
for m = 1 : 2
  fprintf(fids(m), 'Data type   : RealData\n');
  fprintf(fids(m), 'Date created: %s\n\n', datestr(now));
  
  fprintf(fids(m), '======================\n');
  fprintf(fids(m), '            SRMR\n');
  fprintf(fids(m), '----------------------\n');
  fprintf(fids(m), '            \t   enh\n');
  fprintf(fids(m), '----------------------\n');
end

for i1 = 1 : length(dists)
  dist = dists{i1};
  name = ['et_', dist, '_room1'];
  
  for m = 1 : 2
    fprintf(fids(m), '%14s\t', name);
  end
  
  for i2 = 1 : length(types)
    typ = types{i2};
    for m = 1 : 2
      fprintf(fids(m), '%6.2f\t', srmr_mean(i1, i2));
    end
  end
  for m = 1 : 2
    fprintf(fids(m), '\n');
  end
end

name = 'average';
for m = 1 : 2
  fprintf(fids(m), '----------------------\n');
  fprintf(fids(m), '%14s\t%6.2f\t\n', ...
	  name, ...
	  mean(srmr_mean(:, 1)));
%	  mean(srmr_mean(:, 2)));
  fprintf(fids(m), '======================\n');  
end  

fclose(fid);

end
