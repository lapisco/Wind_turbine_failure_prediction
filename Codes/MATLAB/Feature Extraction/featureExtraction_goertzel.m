function  GOERTZEL = featureExtraction_goertzel(data, varargin)
% =========================================================================
% A feature extraction module based on Goertzel algorith to get frequencies
% indices of the Fourier transform.
% =========================================================================
% Fucntion call:
% =====================
%
% -- Format 01:
% featureExtraction_goertzel(data, fs, fundamental, harmonics)
%
% =====================
% Arguments:
% ===
% - data: input signal expressed in a 1xn vector
% - fs: sampling frequency 
% - fundamental: fundamental frequency
% - harmonics: n multiples of fundamental frequencies, wherein n is
% 1,2,3..n_max
% - windowSize: is the windows size to look for harmonic values. By default
% is set as 1 Hz.
% =====================
% Returns:
% ===
% - GOERTZEL: a struct variable, with the following fields:
% - frequ_indices: indices in the vector that represent the harmonics you 
% want to get;
% - harmonics.f: frequencies returned according harmonics argument;
% - harmonics.value: amplitude values returned according harmonics argument.
% =========================================================================
% Author: Navar M M N
% contact: navarmedeiros@gmail.com
% last update: 22/10/2017
% ps.: Please, let me know any bugs you may find, or suggetions in order to
% upgrade our work.

%% GOERTZEL process:

%%    
if nargin >= 4
  
   
   fs = varargin{1};
   fund = varargin{2};
   harmonics = varargin{3};
   
   freq_indices = round((harmonics.*fund)/fs*(length(data)-1)) + 1;

   GOERTZEL.harmonics.values = abs(goertzel(data, freq_indices)/length(data));
   
   GOERTZEL.freq_indices = freq_indices; 
   
   GOERTZEL.harmonics.f = harmonics.*fund;
   
end


