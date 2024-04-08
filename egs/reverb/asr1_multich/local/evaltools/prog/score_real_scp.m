%% score_real.m
%%
%% Evaluate speech enhancement results for RealData. This is
%% designed for use in the REVERB challenge.
%%
%% Written and distributed by the REVERB challenge organizers on 1 July, 2013
%% Inquiries to the challenge organizers (REVERB-challenge@lab.ntt.co.jp)



function score_real_scp(name, resdir, tgtlist, tgtroot, srmrdir)
% name et
% resdir result file
% tgtlist a .scp file that contains file names and their paths
% srmrdir

% Set up
%----------------------------------------------------------------------

addpath(srmrdir);
warning('off', 'MATLAB:dispatcher:pathWarning');

resdir = fullfile(resdir, 'work');

cmd = ['mkdir -p -v ', resdir];
system(cmd);

% Analyze name
% name = ['et_', dist, '_room1_senh'];
%----------------------------------------------------------------------
tmp = strsplit(name,'_');
dt_or_et = tmp(1);
near_or_far = tmp(2);

% List files to be evaluated.
%----------------------------------------------------------------------

num_file = 0;
tgt_name = cell(10000, 1);
tgt      = cell(10000, 1);
tgtfid   = fopen(tgtlist);

while ~feof(tgtfid)
  line = fgetl(tgtfid);
  newline = strsplit(line);
  tmpname = strtrim(newline(1));
  if ~isempty(strfind(tmpname{1}, near_or_far{1}))
    num_file = num_file + 1;
    tgt_name{num_file} = tmpname;
    tgt{num_file} = strtrim(newline(2));
  elseif ~isempty(strfind(tmpname{1}, 'LA1'))
    num_file = num_file + 1;
    tgt_name{num_file} = tmpname;
    tgt{num_file} = strtrim(newline(2));
  % process DIRHA dataset
  end
end

fclose(tgtfid);

tgt_name = tgt_name(1 : num_file);
tgt = tgt(1 : num_file);

% Create a result file.
%----------------------------------------------------------------------

fid  = fopen(fullfile(resdir, name), 'w');
fids = [1, fid];

% Evaluate each file.
%----------------------------------------------------------------------

for m = 1 : 2
  fprintf(fids(m), '%s\n', datestr(now, 'mmmm dd, yyyy  HH:MM:SS AM'));
  fprintf(fids(m), '%s\n\n', fullfile(pwd, mfilename));

  fprintf(fids(m), 'TARGET LIST    : %s\n'  , tgtlist);
  fprintf(fids(m), 'TARGET ROOT    : %s\n'  , tgtroot);

  fprintf(fids(m), 'SRMR directory:\n');
  fprintf(fids(m), '%s\n\n', srmrdir);

  fprintf(fids(m), '----------------------------------------------------------------------\n');
  fprintf(fids(m), 'Individual results\n\n');
end

srmr_mean = zeros(num_file, 1);

for k = 1 : num_file
  tgtname = fullfile(tgtroot, tgt{k});

%  tgtname = fullfile(tgt{k});
  for m = 1 : 2
    fprintf(fids(m), '[%04d of %04d]\n', k, num_file);
    fprintf(fids(m), 'TARGET   : %s\n' , tgtname{1});
  end

  %%%%%%%%%%
  %% SRMR %%
  %%%%%%%%%%
  [y,Fs] = audioread(tgtname{1});
  srmr_mean(k) = SRMR(y(:,1), Fs);

  for m = 1 : 2
    fprintf(fids(m), '\tSRMR           : %6.2f\n', srmr_mean(k));
  end
end


% Print a summary.
%----------------------------------------------------------------------

avg_srmr_mean = mean(srmr_mean);

for m = 1 : 2
  fprintf(fids(m), '----------------------------------------------------------------------\n');
  fprintf(fids(m), 'Summary\n\n');
  fprintf(fids(m), 'AVG SRMR           : %6.2f\n', avg_srmr_mean);
end

fclose(fid);
