%% score_SimData.m
%%
%% Evaluate speech enhancement results for SimData. This is
%% designed for use in the REVERB challenge.
%%
%% Written and distributed by the REVERB challenge organizers on 1 July, 2013
%% Inquiries to the challenge organizers (REVERB-challenge@lab.ntt.co.jp)



function score_STOI_scp(download_from_ldc,cln_scp_file,senhroot,scp_file,pesqdir,compute_pesq)

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

    name = ['et_', dist, '_', room, '_senh'];

    tmp = strsplit(name,'_');
    dt_or_et = tmp(1);
    near_or_far = tmp(2);
    room_id = tmp(3);

    tgtlist = scp_file;
    reflist = cln_scp_file;

    tgt = cell(10000, 1);
    tgt_name = cell(10000, 1);
    ref = cell(10000, 1);
    ref_name = cell(10000, 1);

    ref_sel = cell(10000, 1);
    ref_name_sel = cell(10000, 1);

    num_file = 0;
    num_file_ref = 0;

    tgtfid = fopen(tgtlist);
    reffid = fopen(reflist);

    while ~feof(reffid)
        line = fgetl(reffid);
        num_file_ref = num_file_ref + 1;
        newline = strsplit(line);
        ref_name{num_file_ref} = strtrim(newline(1));
        ref{num_file_ref} = strtrim(newline(2));
    end
    ref_name = ref_name(1 : num_file_ref);
    ref = ref(1 : num_file_ref);
    fclose(reffid);

    refer_name = ['et_', dist, '_', room, '_ref.scp'];
    target_name = ['et_', dist, '_', room, '_enh.scp'];
    ref_scp = fopen(refer_name,'w');
    tar_scp = fopen(target_name,'w');

    while ~feof(tgtfid)
      line = fgetl(tgtfid);
      newline = strsplit(line);
      tmpname = strtrim(newline(1));
      if ~isempty(strfind(tmpname{1}, near_or_far{1})) && ~isempty(strfind(tmpname{1}, room_id{1}))
          num_file = num_file + 1;
          tgt_name{num_file} = tmpname;
          tgt{num_file} = strtrim(newline(2));
          for j = 1 : num_file_ref
              if strcmp(ref_name{j}, strrep(tmpname,strcat('8ch_', near_or_far{1}),'cln'))
                a = ref_name{j};
                b = strrep(tmpname,strcat('8ch_', near_or_far{1}),'cln');
                ref_name_sel{num_file} = ref_name{j};
                ref_sel{num_file} = ref{j};
                break
              end
          end
	  newpath = tgt{num_file};
          fprintf(tar_scp, '%s %s\n', tmpname{1}, fullfile(senhroot , newpath{1}));
	  newpath = ref_sel{num_file};
          fprintf(ref_scp, '%s %s\n', tmpname{1}, newpath{1});
      end

    end

    fclose(tar_scp);
    fclose(ref_scp);

    fclose(tgtfid);

  end
end

end
