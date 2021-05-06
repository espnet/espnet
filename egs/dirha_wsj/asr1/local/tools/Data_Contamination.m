%%%%%%%%%%%%%%%%%%%%%%
% Author: Mirco Ravanelli (mravanelli@fbk.eu)
%
% NOV 2015
%
% Description:
% The first part of the script (#TRAINING DATA) starts from the standard close-talk version of the wsj dataset and contaminates it with reverberation.
% The second part of this script (#DIRHA DATA) extracts the DIRHA wsj sentences of a given microphone (e.g., LA6, Beam_Circular_Array, L1R, LD07, Beam_Linear_Array, etc.)
% from the available 1-minute sequences. It normalizes the amplitude of each signal and performs a conversion of the xml label into a txt label.
% After running the script, the training and test databases (to be used in the kaldi recipe) will be available in the specified output folder.
%
% Modified from the script for dirha_wsj in Espnet
% Xiaofei Wang 06/01/2019
% Ruizhi Li 09/07/2019
%
%%%%%%%%%%%%%%%%%%%%%%
function Data(mic_sel, wsj1_folder, wsj0_folder, dirha_folder, out_folder, IR_folder, sph_reader)

% mic: Select here one of the available microphone (e.g., LA6, L1R, LD07, Beam_Circular_Array,Beam_Linear_Array, etc. => Please, see Floorplan)
% wsj1_folder: Path of the original close-talk WSJ1 dataset
% wsj0_folder: Path of the original close-talk WSJ0 dataset
% dirha_folder: Path of the DIRHA_data
% out_folder: Path where to store the processed data
% IR_folder: Impulse responses for WSJ contamination
% sph_reader: Directory for the sphere reader.  It should be downloaded https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools and compiled


warning off


% Parameters to set
%-----------------------------------------------------------------------

wsj_folders={wsj1_folder,wsj0_folder}; % Path of the original close-talk WSJ1/WSJ0 dataset
wsj_names={'WSJ1_contaminated_mic', 'WSJ0_contaminated_mic'};  %name of the output contaminated WSJ folder
dirha_name='DIRHA_wsj_oracle_VAD_mic'; %name of the output DIRHA_dataset folder (with an Oracle VAD applied to the 1-minute sequences)

% Impulse responses for WSJ contamination
IR_folders{1}=strcat(IR_folder, '/T1_O6');
IR_folders{2}=strcat(IR_folder, '/T2_O5');
IR_folders{3}=strcat(IR_folder, '/T3_O3');

%-----------------------------------------------------------------------

% Check version of MATLAB
vers=version('-release');
vers=str2double(vers(1:end-1));

% Creation of the output folder
mkdir(out_folder);


%%%%  TRAINING DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for idx = 1 : 2

 wsj_folder = wsj_folders{idx};
 wsj_name = wsj_names{idx};

 mkdir(strcat(out_folder,'/',wsj_name,'_',mic_sel));

 fprintf('-----------------------------\n');
 fprintf('Contamination of the WSJ database using mic %s:\n',mic_sel);


 Norm_factor=80000; % normalization factor

 % Generation of a WSJ folder with the same structure of the original WSJ folder
 fprintf('Folders creation...\n');

 wsj_folder_clean=strcat(wsj_folder);

 create_folder_str(wsj_folder_clean,strcat(out_folder,'/',wsj_name,'_',mic_sel));

 % copy of the transcription files
 if idx == 1
  % wsj1
  out_dir = strcat(out_folder,'/',wsj_name,'_',mic_sel,'/13-32.1/wsj1/doc')
  mkdir out_dir
  copyfile(strcat(wsj_folder,'/13-32.1/wsj1/doc'), out_dir);
  out_dir = strcat(out_folder,'/',wsj_name,'_',mic_sel,'/13-34.1/wsj1/doc')
  mkdir out_dir
  copyfile(strcat(wsj_folder,'/13-34.1/wsj1/doc'), out_dir);
  out_dir = strcat(out_folder,'/',wsj_name,'_',mic_sel,'/13-34.1/wsj1/trans')
  mkdir out_dir
  copyfile(strcat(wsj_folder,'/13-34.1/wsj1/trans'), out_dir);
 else
  % wsj0
  out_dir = strcat(out_folder,'/',wsj_name,'_',mic_sel,'/11-13.1/wsj0/doc')
  mkdir out_dir
  copyfile(strcat(wsj_folder,'/11-13.1/wsj0/doc'), out_dir);
  out_dir = strcat(out_folder,'/',wsj_name,'_',mic_sel,'/11-4.1/wsj0/transcrp')
  mkdir out_dir
  copyfile(strcat(wsj_folder,'/11-4.1/wsj0/transcrp'), out_dir);
 end

 % list of all the original WSJ files
 list=find_files(wsj_folder_clean,'.wv1');

 count=0;

 for i=1:length(list)

  count=count+1;

  % loading the training impulse responses
  IR_file=strcat(IR_folders{count},'/',mic_sel);
  load(IR_file);

  % Reading the original WSJ signal
  signal=read_sphere(sph_reader,list{i});

  % 16-48 kHz conversion (IRs were measured at 48 kHz)
  signal=resample(signal,3,1);
  signal=signal./max(abs(signal));

  % Convolution
  signal_rev=fftfilt(risp_imp,signal);
  signal_rev=signal_rev/Norm_factor;

  % Compensation for the delay due to time of Flight (ToF)
  [v, p]=max(risp_imp);
  signal_rev=linear_shift(signal_rev',-p);

  % 48-16 kHz conversion
  signal_rev=resample(signal_rev,1,3);
  signal_rev=signal_rev./max(abs(signal_rev));

  % saving the output wavfile
  name_wav=strrep(list{i},wsj_folder_clean,strcat(out_folder,'/',wsj_name,'_',mic_sel));
  name_wav=strrep(name_wav,'.wv1','.wav');

  if vers>=2014
   audiowrite(name_wav,0.95.*signal_rev,16000)
  else
   wavwrite(0.95.*signal_rev,16000,name_wav)
  end

  fprintf('done %i/%i %s\n',i,length(list),name_wav);

  % change impulse response
  if count==length(IR_folders)
  count=0;
  end

 end
end



%%% DIRHA DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%list of all the files for the specified microphone
list=find_files(dirha_folder,strcat(mic_sel,'.wav'));


fprintf('-----------------------------\n');
fprintf('Extraction of the DIRHA_wsj database for mic %s:\n',mic_sel);

for i=1:length(list)

 % opening the original annotation file
 xml_file=strrep(list{i},'.wav','.xml');

 fid = fopen(xml_file);
 tline = fgetl(fid);

 while ischar(tline)


    if isempty(strfind(tline,'<name>'))==0

    % source id extraction
    find1=strfind(tline,'>');
    find2=strfind(tline,'<');
    source_id=tline(find1(1)+1:find2(2)-1);

    % spk_id extraction
    tline = fgetl(fid);
    find1=strfind(tline,'>');
    find2=strfind(tline,'<');
    spk_id=tline(find1(1)+1:find2(2)-1);

    % spk_info extraction
    tline = fgetl(fid);
    find1=strfind(tline,'>');
    find2=strfind(tline,'<');
    gender=tline(find1(1)+1:find2(2)-1);

    % begin sentence extraction
    tline = fgetl(fid);
    tline = fgetl(fid);
    find1=strfind(tline,'>');
    find2=strfind(tline,'<');
    beg_mix=str2double(tline(find1(1)+1:find2(2)-1));

    % end sentence extraction
    tline = fgetl(fid);
    find1=strfind(tline,'>');
    find2=strfind(tline,'<');
    end_mix=str2double(tline(find1(1)+1:find2(2)-1));


    % creation of the output folder
    if isempty(strfind(list{i},'/Sim/'))==0
    data_type='Sim';
    else
    data_type='Real';
    end

    folder_spk=strcat(out_folder,'/',dirha_name,'_',mic_sel,'/',data_type,'/',gender,'/',spk_id);
    mkdir(folder_spk);

    % sentence extraction in the 1-minute sequence
    if vers>=2014
     signal=audioread(list{i},[beg_mix end_mix]);
    else
     signal=wavread(list{i},[beg_mix end_mix]);
    end

    % amplutute normalization
    signal=0.95.*signal./max(abs(signal));

    % saving the extracted sentence
    name_wav=strcat(folder_spk,'/',source_id,'.wav');
    if vers>=2014
     audiowrite(name_wav,signal,16000);
    else
     wavwrite(0.95.*signal,16000,name_wav)
    end

    % txt label extraction
    tline = fgetl(fid);
    find1=strfind(tline,'>');
    find2=strfind(tline,'<');
    text=tline(find1(1)+1:find2(2)-1);

    % write the label in a txt file
    name_txt=strrep(name_wav,'.wav','.txt');
    fid_w=fopen(name_txt,'w');
    fprintf(fid_w,'0 %i %s\n',length(signal)-1,text);
    fclose(fid_w);

    fprintf('done %s\n',name_wav);
    end

    tline = fgetl(fid);
 end

 fclose(fid);

end

fprintf('-----------------------------\n');
fprintf('DONE!\n')
fprintf('Extracted data available here: \n wsj1_contaminated= %s\n wsj0_contaminated= %s\n dirha= %s\n',strcat(out_folder,'/',wsj_names{1},'_',mic_sel),strcat(out_folder,'/',wsj_names{2},'_',mic_sel),strcat(out_folder,'/',dirha_name,'_',mic_sel)');
end
