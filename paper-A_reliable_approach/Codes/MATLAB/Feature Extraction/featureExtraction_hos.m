function HOS = featureExtraction_hos(data, varargin)
% =========================================================================
% A feature extraction module based on Higher Order Statistics (HOS).
% =========================================================================
% Fucntion call:
% =====================
% -- Format 01:
% featureExtraction_hos(data)
%
% -- Format 02:
% featureExtraction_hos(data, moments)
%
% =====================
% Arguments:
% ===
% - data: input signal expressed in a 1xn vector or a nxn matrix
% - moments: a cell string containing name of statistical measures you
% want to get. For now there are only 'rms' and 'variance'
% =====================
% Returns:
% ===
% - HOS: a struct variable, with the following fields:
% - skewness:
% - kurtosis: 
% - variance:
% - rms: 
% =========================================================================
% Author: Navar M M N
% contact: navarmedeiros@gmail.com
% last update: 22/10/2017
% ps.: Please, let me know any bugs you may find, or suggetions in order to
% upgrade our work.
% =========================================================================


HOS.skewness = skewness(data);
HOS.kurtosis = kurtosis(data);

if nargin == 2
    
    moments = varargin{1};
    
    for  i = 1:length(moments)
        switch moments{i} 
            case 'rms'
                HOS.rms = rms(data);
            case 'variance'
                HOS.variance = var(data);
        end
    end
        
end

end

