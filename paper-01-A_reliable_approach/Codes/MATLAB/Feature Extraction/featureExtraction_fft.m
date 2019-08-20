function  FFT = featureExtraction_fft(data, varargin)
% =========================================================================
% A feature extraction module based on Fast Fourier Transform.
% =========================================================================
% Fucntion call:
% =====================
% -- Format 01:
% featureExtraction_fft(data)
%
% -- Format 02:
% featureExtraction_fft(data, fs)
%
% -- Format 03:
% featureExtraction_fft(data, fs, fundamental, harmonics)
%
% -- Format 04:
% featureExtraction_fft(data, fs, fundamental, harmonics, windowSize)
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
% - FFT: a struct variable, with the following fields:
% - magnitude: a 1xn magnitude vector, containg amplitude of each harmonic
% presented in data signal. Its return respect Nyquidst Theorem.
% - frequency: a frequency vector, based in sampling frequency (fs).
% - harmonics.f: frequencies returned according harmonics argument.
% - harmonics.value: amplitude values returned according harmonics argument.
% =========================================================================
% Author: Navar M M N
% contact: navarmedeiros@gmail.com
% last update: 22/10/2017
% ps.: Please, let me know any bugs you may find, or suggetions in order to
% upgrade our work.

%% FFT process:

Y = fft(data);
L = length(Y);

Mag = abs(Y/L);
Mag_final = Mag(1:fix(L/2)+1);
Mag_final = Mag_final * 2;

FFT.magnitude = Mag_final;

%% 

if nargin == 2
   
   fs = varargin{1};
   FFT.frequency = (fs*(0:(L/2))/L)'; 
   
elseif nargin >= 4
   
   fs = varargin{1};
   fund = varargin{2};
   harmonics = varargin{3};
   
   FFT.frequency = (fs*(0:(L/2))/L)'; 
   res = fs/L;
   
   FFT.harmonics.f = harmonics.*fund;
   idx = ceil(FFT.harmonics.f./res);
   idx(1) = idx(1) + 1;
   
   if nargin == 4
       windowSize = 1;
       
       for i = 1:length(idx)
           
           range = ceil(windowSize/res);
           buff = FFT.magnitude(idx(i)-range:idx(i)+range);
           
           FFT.harmonics.values(i,1) = max(buff);
           %FFT.harmonics.values = FFT.magnitude(idx);  
           
       end
   elseif nargin == 5
       
       for i = 1:length(idx)
           windowSize = varargin{4};
           
           range = ceil(windowSize/res);
           buff = FFT.magnitude(idx(i)-range:idx(i)+range);
           
           FFT.harmonics.values(i,1) = max(buff);
           %FFT.harmonics.values = FFT.magnitude(idx);  
           
       end
   end
end

end

