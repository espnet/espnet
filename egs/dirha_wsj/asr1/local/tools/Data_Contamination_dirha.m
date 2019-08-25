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
%%%%%%%%%%%%%%%%%%%%%%

clear
clc
close all

warning off

% Parameters to set
%-----------------------------------------------------------------------

% Paths of the original datasets
wsj_folder='/export/corpora4/CHiME4/CHiME3/data/WSJ0/wsj0'; % Path of the original close-talk WSJ dataset
dirha_folder='/export/b18/xwang/data/dirha_english_wsj_audio'; % Path of the DIRHA_data

% output paths/names
out_folder='/export/b08/ruizhili/data/Data_processed'; % Path where to store the processed data

%wsj_name='WSJ0_contaminated_mic';  %name of the output contaminated WSJ folder
dirha_name='DIRHA_wsj_oracle_VAD_mic'; %name of the output DIRHA_dataset folder (with an Oracle VAD applied to the 1-minute sequences)

% Selected microphone
mic_sels={'Beam_Circular_Array','Beam_Linear_Array', 'L1C', 'KA6'}; % Select here one of the available microphone (e.g., LA6, L1R, LD07, Beam_Circular_Array,Beam_Linear_Array, etc. => Please, see Floorplan)

% Impulse responses for WSJ contamination (Default is ../Training_IRs/*)
IR_folder{1}='/export/b18/xwang/data/DIRHA_English_phrich_released_june2016_realonly_last/Data/Training_IRs/T1_06';
IR_folder{2}='/export/b18/xwang/data/DIRHA_English_phrich_released_june2016_realonly_last/Data/Training_IRs/T2_05';
IR_folder{3}='/export/b18/xwang/data/DIRHA_English_phrich_released_june2016_realonly_last/Data/Training_IRs/T3_03';

%-----------------------------------------------------------------------

% Check version of MATLAB
vers=version('-release');
vers=str2double(vers(1:end-1));


for mmm = 1 : length(mic_sels)

	mic_sel = mic_sels{mmm};

% Creation of the output folder
mkdir(out_folder); 
%mkdir(strcat(out_folder,'/',wsj_name,'_',mic_sel));



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
%fprintf('Extracted data available here: \n wsj_contaminated= %s\n dirha= %s\n',strcat(out_folder,'/',wsj_name,'_',mic_sel),strcat(out_folder,'/',dirha_name,'_',mic_sel)');

end

