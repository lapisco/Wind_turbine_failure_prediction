%% Script to generate a data set:
% =============================================
% ===== Database information:
% SCIG Generator 
% Normal
% Short-Circuit Motor
% 5000 hz, 10 s
% ===== Features:
% SCM props
% =============================================
% NAME: 
% ===== Date: 25/10/2017
%%
clc; clear; close all
addpath('../../'); addpath('../../filters/');
%%
% Open data:
filename = 'v000_NORMAL_FR4500_FG4385_L000_1,0IN_SENSORC.csv'

Data = readtable(filename);

%% Params feature extraction with SCM
% Input signal filtered:

params.Window_size        = 7;
params.Filter_type{1}     = 'movAvg';

Data.Current_R_filtered = filter_signal(Data.Current_R, params);

% windowSize = 5; 
% b = (1/windowSize)*ones(1,windowSize)

%%          

SCM = featureExtraction_scm(Data.Current_R, Data.Current_R_filtered);

%%

disp('=================')
disp('Features:')
disp('=================')

SCM
