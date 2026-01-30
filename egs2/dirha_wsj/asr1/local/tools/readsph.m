function [y,fs,wrd,phn,ffx]=readsph(filename,mode,nmax,nskip)
%READSPH  Read a SPHERE/TIMIT format sound file [Y,FS,WRD,PHN,FFX]=(FILENAME,MODE,NMAX,NSKIP)
%
% Input Parameters:
%
%	FILENAME gives the name of the file (with optional .SPH extension) or alternatively
%                 can be the FFX output from a previous call to READSPH having the 'f' mode option
%	MODE		specifies the following (*=default):
%
%    Scaling: 's'    Auto scale to make data peak = +-1 (use with caution if reading in chunks)
%             'r'    Raw unscaled data (integer values)
%             'p' *	Scaled to make +-1 equal full scale
%             'o'    Scale to bin centre rather than bin edge (e.g. 127 rather than 127.5 for 8 bit values)
%                     (can be combined with n+p,r,s modes)
%             'n'    Scale to negative peak rather than positive peak (e.g. 128.5 rather than 127.5 for 8 bit values)
%                     (can be combined with o+p,r,s modes)
%   Format    'l'    Little endian data (Intel,DEC) (overrides indication in file)
%             'b'    Big endian data (non Intel/DEC) (overrides indication in file)
%
%   File I/O: 'f'    Do not close file on exit
%             'd'    Look in data directory: voicebox('dir_data')
%             'w'    Also read the annotation file *.wrd if present (as in TIMIT)
%             't'    Also read the phonetic transcription file *.phn if present (as in TIMIT)
%                    Eac line of the annotation and transcription files is of the form: m n token
%                    where m and n are start end end times in samples and token is a word or phoneme test descriptor
%                    The corresponding cell arrays WRD and PHN contain two elements per row: {[m n]/fs 'token'}
%                    These outputs are only present if the corresponding 'w' and 't' options are selected
%
%	NMAX     maximum number of samples to read (or -1 for unlimited [default])
%	NSKIP    number of samples to skip from start of file
%               (or -1 to continue from previous read when FFX is given instead of FILENAME [default])
%
% Output Parameters:
%
%	Y          data matrix of dimension (samples,channels)
%	FS         sample frequency in Hz
%	WRD{*,2}   cell array with word annotations: WRD{*,:)={[t_start t_end],'text'} where times are in seconds
%              only present if 'w' option is given
%	PHN{*,2}   cell array with phoneme annotations: PHN{*,:)={[t_start	t_end],'phoneme'} where times are in seconds
%              only present if 't' option is present
%	FFX        Cell array containing
%
%     {1}     filename
%     {2}     header information
%        {1}  first header field name
%        {2}  first header field value
%     {3}     format string (e.g. NIST_1A)
%     {4}(1)  file id
%        (2)  current position in file
%        (3)  dataoff	byte offset in file to start of data
%        (4)  order  byte order (l or b)
%        (5)  nsamp	number of samples
%        (6)  number of channels
%        (7)  nbytes	bytes per data value
%        (8)  bits	number of bits of precision
%        (9)  fs	sample frequency
%		 (10) min value
%        (11) max value
%        (12) coding: 0=PCM,1=uLAW + 0=no compression,10=shorten,20=wavpack,30=shortpack
%        (13) file not yet decompressed
%     {5}     temporary filename
%
%   If no output parameters are specified, header information will be printed.
%   To decode shorten-encoded files, the program shorten.exe must be in the same directory as this m-file
%
%  Usage Examples:
%
% (a) Draw an annotated spectrogram of a TIMIT file
%           filename='....TIMIT/TEST/DR1/FAKS0/SA1.WAV';
%           [s,fs,wrd,phn]=readsph(filename,'wt');
%           spgrambw(s,fs,'Jwcpta',[],[],[],[],wrd);

%	   Copyright (C) Mike Brookes 1998
%      Version: $Id: readsph.m 713 2011-10-16 14:45:43Z dmb $
%
%   VOICEBOX is a MATLAB toolbox for speech processing.
%   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   http://www.gnu.org/copyleft/gpl.html or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

persistent BYTEORDER
codes={'sample_count'; 'channel_count';  'sample_n_bytes';'sample_sig_bits'; 'sample_rate'; 'sample_min'; 'sample_max'};
codings={'pcm'; 'ulaw'};
compressions={',embedded-shorten-';',embedded-wavpack-'; ',embedded-shortpack-'};
if isempty(BYTEORDER), BYTEORDER='l'; end
if nargin<1, error('Usage: [y,fs,hdr,fidx]=READSPH(filename,mode,nmax,nskip)'); end
if nargin<2, mode='p';
else mode = [mode(:).' 'p'];
end
k=find((mode>='p') & (mode<='s'));
mno=all(mode~='o');                      % scale to input limits not output limits
sc=mode(k(1));
if any(mode=='l'), BYTEORDER='l';
elseif any(mode=='b'), BYTEORDER='b';
end
if nargout
    ffx=cell(5,1);
    if ischar(filename)
        if any(mode=='d')
            filename=fullfile(voicebox('dir_data'),filename);
        end
        fid=fopen(filename,'rb',BYTEORDER);
        if fid == -1
            fn=[filename,'.sph'];
            fid=fopen(fn,'rb',BYTEORDER);
            if fid ~= -1, filename=fn; end
        end
        if fid == -1
            error('Can''t open %s for input',filename);
        end
        ffx{1}=filename;
    else
        if iscell(filename)
            ffx=filename;
        else
            fid=filename;
        end
    end

    if isempty(ffx{4});
        fseek(fid,0,-1);
        str=char(fread(fid,16)');
        if str(8) ~= 10 || str(16) ~= 10, fclose(fid); error('File does not begin with a SPHERE header'); end
        ffx{3}=str(1:7);
        hlen=str2double(str(9:15));
        hdr={};
        while 1
            str=fgetl(fid);
            if str(1) ~= ';'
                [tok,str]=strtok(str);
                if strcmp(tok,'end_head'), break; end
                hdr(end+1,1)={tok};
                [tok,str]=strtok(str);
                if tok(1) ~= '-', error('Missing ''-'' in SPHERE header'); end
                if tok(2)=='s'
                    hdr(end,2)={str(2:str2num(tok(3:end))+1)};
                elseif tok(2)=='i'
                    hdr(end,2)={sscanf(str,'%d',1)};
                else
                    hdr(end,2)={sscanf(str,'%f',1)};
                end
            end
        end
        i=find(strcmp(hdr(:,1),'sample_byte_format'));
        if ~isempty(i)
            bord=char('b'+('l'-'b')*(hdr{i,2}(1)=='0'));
            if bord ~= BYTEORDER && mode~='b' && mode ~='l'
                BYTEORDER=bord;
                fclose(fid);
                fid=fopen(filename,'rb',BYTEORDER);
            end
        end
        i=find(strcmp(hdr(:,1),'sample_coding'));
        icode=0;                % initialize to PCM coding
        if ~isempty(i)
            icode=-1;                   % unknown code
            scode=hdr{i,2};
            nscode=length(scode);
            for j=1:length(codings)
                lenj=length(codings{j});
                if strcmp(scode(1:min(nscode,lenj)),codings{j})
                    if nscode>lenj
                        for k=1:length(compressions)
                            lenk=length(compressions{k});
                            if strcmp(scode(lenj+1:min(lenj+lenk,nscode)),compressions{k})
                                icode=10*k+j-1;
                                break;
                            end
                        end
                    else
                        icode=j-1;
                    end
                    break;
                end
            end
        end

        info=[fid; 0; hlen; double(BYTEORDER); 0; 1; 2; 16; 1 ; 1; -1; icode];
        for j=1:7
            i=find(strcmp(hdr(:,1),codes{j}));
            if ~isempty(i)
                info(j+4)=hdr{i,2};
            end
        end
        if ~info(5)
            fseek(fid,0,1);
            info(5)=floor((ftell(fid)-info(3))/(info(6)*info(7)));
        end
        ffx{2}=hdr;
        ffx{4}=info;
    end
    info=ffx{4};
    if nargin<4, nskip=info(2);
    elseif nskip<0, nskip=info(2);
    end

    ksamples=info(5)-nskip;
    if nargin>2
        if nmax>=0
            ksamples=min(nmax,ksamples);
        end
    end

    if ksamples>0
        fid=info(1);
        if icode>=10 && isempty(ffx{5}) %#ok<AND2>
            fclose(fid);
            dirt=voicebox('dir_temp');
            [fnp,fnn,fne,fnv]=fileparts(filename);
            filetemp=fullfile(dirt,[fnn fne fnv]);
            cmdtemp=fullfile(dirt,'shorten.bat');               % batch file needed to convert to short filenames
            % if ~exist(cmdtemp,'file')                   % write out the batch file if it doesn't exist
                cmdfid=fopen(cmdtemp,'wt');
                fprintf(cmdfid,'@"%s" -x -a %%1 "%%~s2" "%%~s3"\n',voicebox('shorten'));
                fclose(cmdfid);
            % end
            if exist(filetemp,'file')                          % need to explicitly delete old file since shorten makes read-only
                doscom=['del /f "' filetemp '"'];
                if dos(doscom) % run the program
                    error('Error running DOS command: %s',doscom);
                end
            end
            if floor(icode/10)==1               % shorten
                doscom=['"' cmdtemp '" ' num2str(info(3)) ' "' filename '" "' filetemp '"'];
                %                     fprintf(1,'Executing: %s\n',doscom);
                if dos(doscom) % run the program
                    error('Error running DOS command: %s',doscom);
                end
            else
                error('unknown compression format');
            end
            ffx{5}=filetemp;
            fid=fopen(filetemp,'r',BYTEORDER);
            if fid<0, error('Cannot open decompressed file %s',filetemp); end
            info(1)=fid;                            % update fid
        end
        info(2)=nskip+ksamples;
        pk=pow2(0.5,8*info(7))*(1+(mno/2-all(mode~='n'))/pow2(0.5,info(8)));  % use modes o and n to determine effective peak
        fseek(fid,info(3)+info(6)*info(7)*nskip,-1);
        nsamples=info(6)*ksamples;
        if info(7)<3
            if info(7)<2
                y=fread(fid,nsamples,'uchar');
                if info(12)==1
                    y=pcmu2lin(y);
                    pk=2.005649;
                else
                    y=y-128;
                end
            else
                y=fread(fid,nsamples,'short');
            end
        else
            if info(7)<4
                y=fread(fid,3*nsamples,'uchar');
                y=reshape(y,3,nsamples);
                y=[1 256 65536]*y-pow2(fix(pow2(y(3,:),-7)),24);
            else
                y=fread(fid,nsamples,'long');
            end
        end
        if sc ~= 'r'
            if sc=='s'
                if info(10)>info(11)
                    info(10)=min(y);
                    info(11)=max(y);
                end
                sf=1/max(max(abs(info(10:11))),1);
            else sf=1/pk;
            end
            y=sf*y;
        end
        if info(6)>1, y = reshape(y,info(6),ksamples).'; end
    else
        y=[];
    end

    if mode~='f'
        fclose(fid);
        info(1)=-1;
        if ~isempty(ffx{5})
            doscom=['del /f ' ffx{5}];
            if dos(doscom) % run the program
                error('Error running DOS command: %s',doscom);
            end
            ffx{5}=[];
        end
    end
    ffx{4}=info;
    fs=info(9);
    wrd=ffx;        % copy ffx into the other arguments in case 'w' and/or 't' are not specified
    phn=ffx;
    if any(mode=='w')
        wrd=cell(0,0);
        fidw=fopen([filename(1:end-3) 'wrd'],'r');
        if fidw>0
            while 1
                tline = fgetl(fidw); % read an input line
                if ~ischar(tline)
                    break
                end
                [wtim, ntim, ee, nix] = sscanf(tline,'%d%d',2);
                if ntim==2
                    wrd{end+1,1}=wtim(:)'/fs;
                    wrd{end,2}=strtrim(tline(nix:end));
                end
            end
            fclose(fidw);
        end
    end
    if any(mode=='t')
        ph=cell(0,0);
        fidw=fopen([filename(1:end-3) 'phn'],'r');
        if fidw>0
            while 1
                tline = fgetl(fidw); % read an input line
                if ~ischar(tline)
                    break
                end
                [wtim, ntim, ee, nix] = sscanf(tline,'%d%d',2);
                if ntim==2
                    ph{end+1,1}=wtim(:)'/fs;
                    ph{end,2}=strtrim(tline(nix:end));
                end
            end
            fclose(fidw);
        end
        if any(mode=='w')
            phn=ph;             % copy into 4th argument
        else
            wrd=ph;             % copy into 3rd argument
        end
    end
else
    [y1,fs,ffx]=readsph(filename,mode,0);
    info=ffx{4};
    if ~isempty(ffx{1}), fprintf(1,'Filename: %s\n',ffx{1}); end
    fprintf(1,'Sphere file type: %s\n',ffx{3});
    fprintf(1,'Duration = %ss: %d channel * %d samples @ %sHz\n',sprintsi(info(5)/info(9)),info(6),info(5),sprintsi(info(9)));
end
