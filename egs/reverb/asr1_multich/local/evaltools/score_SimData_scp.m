%% score_SimData.m
%%
%% Evaluate speech enhancement results for SimData. This is
%% designed for use in the REVERB challenge.
%%
%% Written and distributed by the REVERB challenge organizers on 1 July, 2013
%% Inquiries to the challenge organizers (REVERB-challenge@lab.ntt.co.jp)



function score_SimData_scp(download_from_ldc,cln_scp_file,senhroot,scp_file,pesqdir,compute_pesq)

if ~exist('download_from_ldc', 'var')
  fprintf('Uncomment download_from_ldc!\n');
  return;
end


% Evaluation
%----------------------------------------------------------------------

addpath ./prog;
srmrdir = 'SRMRToolbox';
addpath(genpath('SRMRToolbox/libs'));
pesqexe = 'PESQ';

dists = {'far', 'near'};
rooms = {'room1', 'room2', 'room3'};

taskdir = '../taskfiles/1ch';

%origroot = fullfile(download_from_ldc, 'REVERB_WSJCAM0_et/data');
resdir   = '../scores/SimData';

if exist('pesqdir', 'var') && compute_pesq~=0
  pesqname = fullfile(pesqdir, pesqexe);
else
  pesqname = [];
end

for i1 = 1 : length(dists)
  for i2 = 1 : length(rooms)
    dist = dists{i1};
    room = rooms{i2};

    tgtlist = scp_file;
    reflist = cln_scp_file;

    %% Evaluate the quality of original data
%    name = ['et_', dist, '_', room, '_orig'];
%    score_sim(name, resdir, tgtlist, reflist, origroot, origroot, srmrdir, pesqname);

    %% Evaluate the quality of enhanced data
    name = ['et_', dist, '_', room, '_senh'];
    score_sim_scp(name, resdir, tgtlist, reflist, senhroot, srmrdir, pesqname);
  end
end


% Creating summary
%----------------------------------------------------------------------

types = {'senh'};

workdir = fullfile(resdir, 'work');

cd_mean   = zeros(length(dists), length(rooms), length(types));
cd_med    = zeros(length(dists), length(rooms), length(types));
srmr_mean = zeros(length(dists), length(rooms), length(types));
llr_mean  = zeros(length(dists), length(rooms), length(types));
llr_med   = zeros(length(dists), length(rooms), length(types));
snr_mean  = zeros(length(dists), length(rooms), length(types));
snr_med   = zeros(length(dists), length(rooms), length(types));

if exist('pesqdir', 'var')
  pesq_mean = zeros(length(dists), length(rooms), length(types));
end

for i1 = 1 : length(dists)
  for i2 = 1 : length(rooms)
    for i3 = 1 : length(types)
      dist   = dists{i1};
      room = rooms{i2};
      typ     = types{i3};

      name    = ['et_', dist, '_', room, '_', typ];
      resfile = fullfile(workdir, name);
      fid     = fopen(resfile);
      while ~feof(fid)
	l = fgetl(fid);

	%% CEPSDIST (MEAN)
	if strfind(l, 'AVG CEPSDIST (MEAN)')
	  [nul, l]   = strtok(strtrim(l));
	  [nul, l]   = strtok(strtrim(l));
	  [nul, l]   = strtok(strtrim(l));
	  l          = strtrim(l);
	  if strcmp(l(1), ':')
	    l = strtrim(l(2 : end));
	  end

	  cd_mean(i1, i2, i3) = str2num(strtrim(strtok(l)));
	end

	%% CEPSDIST (MEDIAN)
	if strfind(l, 'AVG CEPSDIST (MED)')
	  [nul, l]   = strtok(strtrim(l));
	  [nul, l]   = strtok(strtrim(l));
	  [nul, l]   = strtok(strtrim(l));
	  l          = strtrim(l);
	  if strcmp(l(1), ':')
	    l = strtrim(l(2 : end));
	  end

	  cd_med(i1, i2, i3) = str2num(strtrim(strtok(l)));
	end

	%% SRMR
	if strfind(l, 'AVG SRMR')
	  [nul, l]   = strtok(strtrim(l));
	  [nul, l]   = strtok(strtrim(l));
	  l          = strtrim(l);
	  if strcmp(l(1), ':')
	    l = strtrim(l(2 : end));
	  end

	  srmr_mean(i1, i2, i3) = str2num(strtrim(strtok(l)));
	end

	%% LLR (MEAN)
	if strfind(l, 'AVG LLR      (MEAN)')
	  [nul, l]   = strtok(strtrim(l));
	  [nul, l]   = strtok(strtrim(l));
	  [nul, l]   = strtok(strtrim(l));
	  l          = strtrim(l);
	  if strcmp(l(1), ':')
	    l = strtrim(l(2 : end));
	  end

	  llr_mean(i1, i2, i3) = str2num(strtrim(strtok(l)));
	end

	%% LLR (MED)
	if strfind(l, 'AVG LLR      (MED)')
	  [nul, l]   = strtok(strtrim(l));
	  [nul, l]   = strtok(strtrim(l));
	  [nul, l]   = strtok(strtrim(l));
	  l          = strtrim(l);
	  if strcmp(l(1), ':')
	    l = strtrim(l(2 : end));
	  end

	  llr_med(i1, i2, i3) = str2num(strtrim(strtok(l)));
	end

	%% FWSEGSNR (MEAN)
	if strfind(l, 'AVG FWSEGSNR (MEAN)')
	  [nul, l]   = strtok(strtrim(l));
	  [nul, l]   = strtok(strtrim(l));
	  [nul, l]   = strtok(strtrim(l));
	  l          = strtrim(l);
	  if strcmp(l(1), ':')
	    l = strtrim(l(2 : end));
	  end

	  snr_mean(i1, i2, i3) = str2num(strtrim(strtok(l)));
	end

	%% FWSEGSNR (MED)
  	if strfind(l, 'AVG FWSEGSNR (MED)')
	  [nul, l]   = strtok(strtrim(l));
	  [nul, l]   = strtok(strtrim(l));
	  [nul, l]   = strtok(strtrim(l));
	  l          = strtrim(l);
	  if strcmp(l(1), ':')
	    l = strtrim(l(2 : end));
	  end

	  snr_med(i1, i2, i3) = str2num(strtrim(strtok(l)));
	end
      end

      %% PESQ
      if exist('pesqdir', 'var')
	if strfind(l, 'AVG PESQ')
	  [nul, l]   = strtok(strtrim(l));
	  [nul, l]   = strtok(strtrim(l));
	  l          = strtrim(l);
	  if strcmp(l(1), ':')
	    l = strtrim(l(2 : end));
	  end

	  pesq_mean(i1, i2, i3) = str2num(strtrim(strtok(l)));
	end
      end

      fclose(fid);
    end
  end
end

fid  = fopen(fullfile(fileparts(resdir), 'score_SimData'), 'w');
fids = [1, fid];

%% Cepstral distance
for m = 1 : 2
  fprintf(fids(m), 'Data type   : SimData\n');
  fprintf(fids(m), 'Date created: %s\n\n', datestr(now));

  fprintf(fids(m), '======================================\n');
  fprintf(fids(m), '           Cepstral distance in dB    \n');
  fprintf(fids(m), '--------------------------------------\n');
  fprintf(fids(m), '            \t  mean\tmedian\n');
  fprintf(fids(m), '--------------------------------------\n');
  fprintf(fids(m), '            \t   enh\t   enh\n');
  fprintf(fids(m), '--------------------------------------\n');
end

for i1 = 1 : length(dists)
  for i2 = 1 : length(rooms)
    dist   = dists{i1};
    room = rooms{i2};
    name    = ['et_', dist, '_', room];

    for m = 1 : 2
      fprintf(fids(m), '%14s\t', name);
    end

    for i3 = 1 : length(types)
      typ = types{i3};
      for m = 1 : 2
	fprintf(fids(m), '%6.2f\t', cd_mean(i1, i2, i3));
      end
    end

    for i3 = 1 : length(types)
      typ = types{i3};
      for m = 1 : 2
	fprintf(fids(m), '%6.2f\t', cd_med(i1, i2, i3));
      end
    end

    for m = 1 : 2
      fprintf(fids(m), '\n');
    end
  end
end
name = 'average';
for m = 1 : 2
  fprintf(fids(m), '--------------------------------------\n');
  fprintf(fids(m), '%14s\t%6.2f\t%6.2f\n', ...
	  name, ...
	  mean(mean(cd_mean(:, :, 1), 1), 2), ...
      mean(mean(cd_med(:, :, 1), 1), 2));
%	  mean(mean(cd_mean(:, :, 2), 1), 2), ...
%	  mean(mean(cd_med(:, :, 1), 1), 2), ...
%	  mean(mean(cd_med(:, :, 2), 1), 2));

  fprintf(fids(m), '======================================\n\n\n');
end


%% SRMR
for m = 1 : 2
  fprintf(fids(m), '======================================\n');
  fprintf(fids(m), '            SRMR  (only mean used)    \n');
  fprintf(fids(m), '--------------------------------------\n');
  fprintf(fids(m), '            \t  mean\tmedian\n');
  fprintf(fids(m), '--------------------------------------\n');
  fprintf(fids(m), '            \t   enh\t   enh\n');
  fprintf(fids(m), '--------------------------------------\n');
end

for i1 = 1 : length(dists)
  for i2 = 1 : length(rooms)
    dist   = dists{i1};
    room = rooms{i2};
    name    = ['et_', dist, '_', room];

    for m = 1 : 2
      fprintf(fids(m), '%14s\t', name);
    end

    for i3 = 1 : length(types)
      typ = types{i3};
      for m = 1 : 2
	fprintf(fids(m), '%6.2f\t', srmr_mean(i1, i2, i3));
      end
    end

    for m = 1 : 2
      fprintf(fids(m), '     -\t     -\t');
    end

    for m = 1 : 2
      fprintf(fids(m), '\n');
    end
  end
end
name = 'average';
for m = 1 : 2
  fprintf(fids(m), '--------------------------------------\n');
  fprintf(fids(m), '%14s\t%6.2f\t      -\n', ...
	  name, ...
      mean(mean(srmr_mean(:, :, 1), 1), 2));
%	  mean(mean(srmr_mean(:, :, 1), 1), 2), ...
%	  mean(mean(srmr_mean(:, :, 2), 1), 2));
  fprintf(fids(m), '======================================\n\n\n');
end


%% LLR
for m = 1 : 2
  fprintf(fids(m), '======================================\n');
  fprintf(fids(m), '             Log likelihood ratio     \n');
  fprintf(fids(m), '--------------------------------------\n');
  fprintf(fids(m), '            \t  mean\tmedian\n');
  fprintf(fids(m), '--------------------------------------\n');
  fprintf(fids(m), '            \t   enh\t   enh\n');
  fprintf(fids(m), '--------------------------------------\n');
end

for i1 = 1 : length(dists)
  for i2 = 1 : length(rooms)
    dist   = dists{i1};
    room = rooms{i2};
    name    = ['et_', dist, '_', room];

    for m = 1 : 2
      fprintf(fids(m), '%14s\t', name);
    end

    for i3 = 1 : length(types)
      typ = types{i3};
      for m = 1 : 2
	fprintf(fids(m), '%6.2f\t', llr_mean(i1, i2, i3));
      end
    end

    for i3 = 1 : length(types)
      typ = types{i3};
      for m = 1 : 2
	fprintf(fids(m), '%6.2f\t', llr_med(i1, i2, i3));
      end
    end

    for m = 1 : 2
      fprintf(fids(m), '\n');
    end
  end
end
name = 'average';
for m = 1 : 2
  fprintf(fids(m), '--------------------------------------\n');
  fprintf(fids(m), '%14s\t%6.2f\t%6.2f\n', ...
	  name, ...
	  mean(mean(llr_mean(:, :, 1), 1), 2), ...
      mean(mean(llr_med(:, :, 1), 1), 2));
%	  mean(mean(llr_mean(:, :, 2), 1), 2), ...
%	  mean(mean(llr_med(:, :, 1), 1), 2), ...
%	  mean(mean(llr_med(:, :, 2), 1), 2));
  fprintf(fids(m), '======================================\n\n\n');
end


%% FWSEGSNR
for m = 1 : 2
  fprintf(fids(m), '======================================\n');
  fprintf(fids(m), 'Frequency-weighted segmental SNR in dB\n');
  fprintf(fids(m), '--------------------------------------\n');
  fprintf(fids(m), '            \t  mean\tmedian\n');
  fprintf(fids(m), '--------------------------------------\n');
  fprintf(fids(m), '            \t   enh\t   enh\n');
  fprintf(fids(m), '--------------------------------------\n');
end

for i1 = 1 : length(dists)
  for i2 = 1 : length(rooms)
    dist   = dists{i1};
    room = rooms{i2};
    name    = ['et_', dist, '_', room];

    for m = 1 : 2
      fprintf(fids(m), '%14s\t', name);
    end

    for i3 = 1 : length(types)
      typ = types{i3};
      for m = 1 : 2
	fprintf(fids(m), '%6.2f\t', snr_mean(i1, i2, i3));
      end
    end

    for i3 = 1 : length(types)
      typ = types{i3};
      for m = 1 : 2
	fprintf(fids(m), '%6.2f\t', snr_med(i1, i2, i3));
      end
    end

    for m = 1 : 2
      fprintf(fids(m), '\n');
    end
  end
end
name = ['average'];
for m = 1 : 2
  fprintf(fids(m), '--------------------------------------\n');
  fprintf(fids(m), '%14s\t%6.2f\t%6.2f\n', ...
	  name, ...
	  mean(mean(snr_mean(:, :, 1), 1), 2), ...
      mean(mean(snr_med(:, :, 1), 1), 2));
%	  mean(mean(snr_mean(:, :, 2), 1), 2), ...
%	  mean(mean(snr_med(:, :, 1), 1), 2), ...
%	  mean(mean(snr_med(:, :, 2), 1), 2));
  fprintf(fids(m), '======================================\n\n\n');
end


%% SRMR
if ~exist('pesqdir', 'var')
  fclose(fid);
  exit;
end

for m = 1 : 2
  fprintf(fids(m), '======================================\n');
  fprintf(fids(m), '            PESQ  (only mean used)    \n');
  fprintf(fids(m), '--------------------------------------\n');
  fprintf(fids(m), '            \t  mean\tmedian\n');
  fprintf(fids(m), '--------------------------------------\n');
  fprintf(fids(m), '            \t   enh\t   enh\n');
  fprintf(fids(m), '--------------------------------------\n');
end

for i1 = 1 : length(dists)
  for i2 = 1 : length(rooms)
    dist   = dists{i1};
    room = rooms{i2};
    name    = ['et_', dist, '_', room];

    for m = 1 : 2
      fprintf(fids(m), '%14s\t', name);
    end

    for i3 = 1 : length(types)
      typ = types{i3};
      for m = 1 : 2
	fprintf(fids(m), '%6.2f\t', pesq_mean(i1, i2, i3));
      end
    end

    for m = 1 : 2
      fprintf(fids(m), '     -\t     -\t');
    end

    for m = 1 : 2
      fprintf(fids(m), '\n');
    end
  end
end
name = 'average';
for m = 1 : 2
  fprintf(fids(m), '--------------------------------------\n');
  fprintf(fids(m), '%14s\t%6.2f\t      -\n', ...
	  name, ...
      mean(mean(pesq_mean(:, :, 1), 1), 2));
%	  mean(mean(pesq_mean(:, :, 1), 1), 2), ...
%	  mean(mean(pesq_mean(:, :, 2), 1), 2));
  fprintf(fids(m), '======================================\n\n\n');
end

fclose(fid);

end
