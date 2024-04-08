function [path,TDOA]=localize(X,chanlist)

% LOCALIZE Tracks the speaker spatial position over time and computes the
% corresponding TDOA using SRP-PHAT and the Viterbi algorithm
%
% [path,TDOA]=localize(X,chanlist)
%
% Inputs:
% Y: nbin x nfram x nchan STFT of the inpu signal
% chanlist: list of input channels (from 1 to 6)
%
% Output:
% path: 3 x nfram position of the speaker over time in centimeters
% TDOA: nchan x nfram corresponding TDOAs between the speaker position and
% the microphone positions
%
% Note: for computational efficiency, the position on the z-axis is assumed
% to be constant over time.
%
% If you use this software in a publication, please cite:
%
% Jon Barker, Ricard Marxer, Emmanuel Vincent, and Shinji Watanabe, The
% third 'CHiME' Speech Separation and Recognition Challenge: Dataset,
% task and baselines, submitted to IEEE 2015 Automatic Speech Recognition
% and Understanding Workshop (ASRU), 2015.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2015-2016 University of Sheffield (Jon Barker, Ricard Marxer)
%                     Inria (Emmanuel Vincent)
%                     Mitsubishi Electric Research Labs (Shinji Watanabe)
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 2,
    chanlist=[1 3:6];
end

% Define hyper-parameters
pow_thresh=-20; % threshold in dB below which a microphone is considered to fail
center_factor=0.05; % weight given to the prior that the speaker's horizontal position is close to the center
smoothing_factor=3; % weight given to the transition probabilities

% Remove zero frequency
% NOTE (Wangyou): also remove the unused channels from X to avoid Error "Index exceeds matrix dimensions" in line 87.
X = X(2:end,:,chanlist);
[nbin,nfram,nchan] = size(X);
wlen=2*nbin;
f=16000/wlen*(1:nbin).';

% Compute relative channel power
if length(chanlist) > 2,
    xpow=shiftdim(sum(sum(abs(X).^2,2),1));
    xpow=10*log10(xpow/max(xpow));
else
    xpow=zeros(1,2);
end

% Define microphone positions in centimeters
xmic=[-10 0 10 -10 0 10]; % left to right axis
ymic=[9.5 9.5 9.5 -9.5 -9.5 -9.5]; % bottom to top axis
zmic=[0 -2 0 0 0 0]; % back to front axis
xmic=xmic(chanlist);
ymic=ymic(chanlist);
zmic=zmic(chanlist);

% Define grid of possible speaker positions in centimeters
xres=46;
xpos=linspace(-45,45,xres);
yres=46;
ypos=linspace(-45,45,yres);
zres=4;
zpos=linspace(15,45,zres);
ngrid=xres*yres*zres;

% Compute horizontal distances between grid points
xvect=reshape(repmat(xpos.',[1 yres]),xres*yres,1);
yvect=reshape(repmat(ypos,[xres 1]),xres*yres,1);
pair_dist=sqrt((repmat(xvect,[1 xres*yres])-repmat(xvect.',[xres*yres 1])).^2+(repmat(yvect,[1 xres*yres])-repmat(yvect.',[xres*yres 1])).^2);

% Compute horizontal distances to the center
center_dist=sqrt((xvect-mean(xpos)).^2+(yvect-mean(ypos)).^2);

% Compute theoretical TDOAs between front pairs
d_grid=zeros(nchan,xres,yres,zres); % speaker-to-microphone distances
for c=1:nchan,
    d_grid(c,:,:,:)=sqrt(repmat((xpos.'-xmic(c)).^2,[1 yres zres])+repmat((ypos-ymic(c)).^2,[xres 1 zres])+repmat((permute(zpos,[3 1 2])-zmic(c)).^2,[xres yres 1]));
end
d_grid=reshape(d_grid,nchan,ngrid);
pairs=[];
for c=1:nchan,
    pairs=[pairs [c*ones(1,nchan-c); c+1:nchan]]; % microphone pairs
end
npairs=size(pairs,2);
tau_grid=zeros(npairs,ngrid); % TDOAs
for p=1:npairs,
    c1=pairs(1,p);
    c2=pairs(2,p);
    tau_grid(p,:)=(d_grid(c2,:)-d_grid(c1,:))/343/100;
end

% Compute the SRP-PHAT pseudo-spectrum
srp=zeros(nfram,ngrid);
for p=1:npairs, % Loop over front pairs
    c1=pairs(1,p);
    c2=pairs(2,p);
    d=sqrt((xmic(c1)-xmic(c2))^2+(ymic(c1)-ymic(c2))^2+(zmic(c1)-zmic(c2))^2);
    alpha=10*343/(d*16000);
    lin_grid=linspace(min(tau_grid(p,:)),max(tau_grid(p,:)),100);
    lin_spec=zeros(nbin,nfram,100); % GCC-PHAT pseudo-spectrum over a uniform interval
    if (xpow(c1)>pow_thresh) && (xpow(c2)>pow_thresh), % discard channels with low power (microphone failure)
        P=X(:,:,c1).*conj(X(:,:,c2));
        P=P./abs(P);
        for ind=1:100,
            EXP=repmat(exp(-2*1i*pi*lin_grid(ind)*f),1,nfram);
            lin_spec(:,:,ind)=ones(nbin,nfram)-tanh(alpha*real(sqrt(2-2*real(P.*EXP))));
        end
    end
    lin_spec=shiftdim(sum(lin_spec,1));
    tau_spec=zeros(nfram,ngrid); % GCC-PHAT pseudo-spectrum over the whole grid
    for t=1:nfram,
        tau_spec(t,:)=interp1(lin_grid,lin_spec(t,:),tau_grid(p,:));
    end
    srp=srp+tau_spec; % sum over the microphone pairs
end

% Loop over possible z-axis positions
path=zeros(zres,nfram);
logpost=zeros(zres,1);
xpath=zeros(zres,nfram);
ypath=zeros(zres,nfram);
zpath=zeros(zres,nfram);
srp=reshape(srp,nfram,xres*yres,zres);
for zind=1:zres,

    % Weight by distance to the center
    weighted_srp=srp(:,:,zind)-center_factor*repmat(center_dist.',[nfram 1]);

    % Track the source position over time
    [path(zind,:),logpost(zind)]=viterbi(weighted_srp.',zeros(xres*yres,1),zeros(xres*yres,1),-smoothing_factor*pair_dist);
    for t=1:nfram,
        [xpath(zind,t),ypath(zind,t)]=ind2sub([xres yres],path(zind,t));
        zpath(zind,t)=zind;
    end
end

% Select the best z-axis position
[~,zind]=max(logpost);
path=(zind-1)*xres*yres+path(zind,:);
xpath=xpos(xpath(zind,:));
ypath=ypos(ypath(zind,:));
zpath=zpos(zpath(zind,:));

% Derive TDOA
d_path=d_grid(:,path);
TDOA=d_path/343/100;
path=[xpath; ypath; zpath];

return
