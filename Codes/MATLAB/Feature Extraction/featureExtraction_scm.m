function  SCM = featureExtraction_scm(data, data_filtered)
% =========================================================================
% A feature extraction module based on SCM transform to get features from
% the co-ocurrence matrix.
% =========================================================================
% Fucntion call:
% =====================
%
% -- Format 01:
% featureExtraction_goertzel(data, data_filtered)
%
% =====================
% Arguments:
% ===
% - data: input signal expressed in a 1xn vector
% - data_filtered: input signal filtered by a low-pass, high-pass, or any 
% filter you desig.
% =====================
% Returns:
% ===
% - SCM: a struct variable, with the fields acoording to scmprops.m:
% =========================================================================
% Author: Navar M M N
% contact: navarmedeiros@gmail.com
% last update: 22/10/2017
% ps.: Please, let me know any bugs you may find, or suggetions in order to
% upgrade our work.

%% SCM process:

%%       
M = scm(data, data_filtered, 'NumParts', 9, 'G',[]);
          
SCM = scmprops(M);


end


