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
% wsj_folder='/export/corpora5/LDC/LDC94S13B'; % Path of the original close-talk WSJ dataset wsj1
wsj_folder='/export/b06/xwang/espnet_e2e/espnet/egs/dirha_wsj/Tools/LDC93S6B';  % wsj0
% output paths/names
out_folder='/export/b08/ruizhili/data/Data_processed'; % Path where to store the processed data

% wsj_name='WSJ1_contaminated_mic';  %name of the output contaminated WSJ folder % wsj1
wsj_name='WSJ0_contaminated_mic';  %name of the output contaminated WSJ folder % wsj1

% Selected microphone

mic_sels={'Beam_Circular_Array','Beam_Linear_Array', 'L1C', 'KA6'};
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
mkdir(strcat(out_folder,'/',wsj_name,'_',mic_sel));




%%%%  TRAINING DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('-----------------------------\n');
fprintf('Contamination of the WSJ database using mic %s:\n',mic_sel);


Norm_factor=80000; % normalization factor

% Generation of a WSJ folder with the same structure of the original WSJ folder
fprintf('Folders creation...\n');

wsj_folder_clean=strcat(wsj_folder);

create_folder_str(wsj_folder_clean,strcat(out_folder,'/',wsj_name,'_',mic_sel));

% copy of the transcription files
%copyfile(strcat(wsj_folder,'/doc'),strcat(out_folder,'/doc'));
%copyfile(strcat(wsj_folder,'/transcrp'),strcat(out_folder,'/transcrp'));


% list of all the original WSJ files
list=find_files(wsj_folder_clean,'.wv1');

count=0;

for i=1:length(list)
    
 count=count+1;
 
 % loading the training impulse responses
 IR_file=strcat(IR_folder{count},'/',mic_sel);
 load(IR_file);

 % Reading the original WSJ signal
 signal=read_sphere(list{i});
 
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
 if count==length(IR_folder)
 count=0;
 end
  
end

fprintf('-----------------------------\n');
fprintf('DONE!\n')
%fprintf('Extracted data available here: \n wsj_contaminated= %s\n dirha= %s\n',strcat(out_folder,'/',wsj_name,'_',mic_sel),strcat(out_folder,'/',dirha_name,'_',mic_sel)');

end

