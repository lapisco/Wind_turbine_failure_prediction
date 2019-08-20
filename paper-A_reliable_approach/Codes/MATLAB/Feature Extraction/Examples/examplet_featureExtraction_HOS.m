%% Script to generate a data set:
% =============================================
% ===== Database information:
% SCIG Generator 
% Normal
% Short-Circuit Motor
% 5000 hz, 10 s
% ===== Features:
% HOS
% =============================================
% NAME: 
% ===== Date: 25/10/2017
%%
clc;
clear all
close all
%%
% Open data:
filename = 'v000_NORMAL_FR4500_FG4385_L000_1,0IN_SENSORC.csv'

Data = readtable(filename);

%% Params feature extraction with SCM

% There are none...

%%          

HOS = featureExtraction_hos(Data.Current_R, {'rms', 'variance'});

%%

disp('=================')
disp('Features:')
disp('=================')

HOS
