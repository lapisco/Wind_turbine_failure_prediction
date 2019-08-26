%% Script to generate a data set:
% =============================================
% ===== Database information:
% SCIG Generator 
% Normal
% Short-Circuit Motor
% 5000 hz, 10 s
% ===== Features:
% Fourier: [0.5, 1, 1.5, 2.5, 3, 5, 7].*f
% Frequency_rotor
% Frequency_gen
% CC_bus
% Power
% Load
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

%% Params feature extraction with Goertzel

fs = 5000;                              % sample rate
harmonics = [1, 0.5, 1.5, 2.5, 3, 5, 7]; % harmonics we desire to get
fund = 43.85;                           % fundamental frequency of the signal

%%

Fourier = featureExtraction_goertzel(Data.Current_R, fs, fund, harmonics)

disp('=================')
disp('Features:')
disp('=================')

Fourier.harmonics.values

