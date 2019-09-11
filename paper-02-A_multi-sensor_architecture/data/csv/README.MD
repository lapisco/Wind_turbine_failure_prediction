Datasets
===

All datasets used in the paper are here. They are

Magnetic axial flux sets:

- [`v000_SCIG_SC_SENSORA_FOURIER.csv`](v000_SCIG_SC_SENSORA_FOURIER.csv)

- [`v000_SCIG_SC_SENSORA_HOS.csv`](v000_SCIG_SC_SENSORA_HOS.csv)

- [`v000_SCIG_SC_SENSORA_SCM.csv`](v000_SCIG_SC_SENSORA_SCM.csv)

Current sets:

- [`v000_SCIG_SC_SENSORC_FOURIER.csv`](v000_SCIG_SC_SENSORC_FOURIER.csv)

- [`v000_SCIG_SC_SENSORC_HOS.csv`](v000_SCIG_SC_SENSORC_HOS.csv)

- [`v000_SCIG_SC_SENSORC_SCM.csv`](v000_SCIG_SC_SENSORC_SCM.csv)

Accelerometer (Vibration) sets:

- [`v000_SCIG_SC_SENSORV_FOURIER.csv`](v000_SCIG_SC_SENSORV_FOURIER.csv)

- [`v000_SCIG_SC_SENSORV_HOS.csv`](v000_SCIG_SC_SENSORV_HOS.csv)

- [`v000_SCIG_SC_SENSORV_SCM.csv`](v000_SCIG_SC_SENSORV_SCM.csv)


The naming convetion follows the rule:

- `v000`: version of the database (concerns only to author).
- `SCIG`: electrical machine is a Squirrel-Cage Induction Generator (SCIG).
- `SC`: The inserted fault is short-circuit (SC)
- `SENSORC`: The sensor used to monitor the machine. 
- `FOURIER`: The feature extracted method used to build the dataset.

The experimental setting is based on [Reboucas *et al*, 2018](https://www.sciencedirect.com/science/article/pii/S0045790618307584) diagram's. More detais are exhibited in this paper.

![Bench](figs/bench.png). 
![Diagram](figs/diagram.png). 


The features in Fourier, HOS and SCM might vary, however the common fields in theses sets are:

## Current sensor (SENSORC)

The common features in files [`v000_SCIG_SC_SENSORC_FOURIER.csv`](v000_SCIG_SC_SENSORC_FOURIER.csv), [`v000_SCIG_SC_SENSORC_HOS.csv`](v000_SCIG_SC_SENSORC_HOS.csv) and [`v000_SCIG_SC_SENSORC_SCM.csv`](v000_SCIG_SC_SENSORC_SCM.csv) are


|    Field   | Description |
|:----------:|:-----------:|
| idx        |    The sample index in dataframe         |
| Freq_Rated |  Frequency applied into Driven Machine's frequency converter (FC-1)             |
| Freq_Gen   |  Frequency applied into Generator's frequency converter (FC-2)           |
| CC_bus     |  Voltage of dc-bus           |
| Power      |  Measures three-phase power           |
| I_R_rms    |  Line Current RMS value           |
| I_S_rms    |  Line Current RMS value           |
| I_T_rms    |  Line Current RMS value           |
| Load       |  Percentage of electrical load in cc-bus           |
| Class      |  Operational condition           |


The unique features in [`v000_SCIG_SC_SENSORC_FOURIER.csv`](v000_SCIG_SC_SENSORC_FOURIER.csv) are:


|    Field   | Description |
|:----------:|:-----------:|
| **FOURIER** | ==================|
| fx1_R      | The fundamental frequency of line current R         |
| fx0d5_R    | The 0.5th multiple of the fundamental frequency  of line current R  |
| fx1d5_R    | The 1.5th multiple of the fundamental frequency  of line current R             |
| fx2d5_R    |   The 2.5th multiple of the fundamental frequency of line current R            |
| fx3_R      |  The 3rd multiple of the fundamental frequency of line current R             |
| fx5_R      |   The 5th multiple of the fundamental frequency of line current R            |
| fx7_R      |    The 7th multiple of the fundamental frequency of line current R   |
| fx1_S      | The fundamental frequency of line current R         |
| fx0d5_S    | The 0.5th multiple of the fundamental frequency  of line current S  |
| fx1d5_S    | The 1.5th multiple of the fundamental frequency  of line current S             |
| fx2d5_S    |   The 2.5th multiple of the fundamental frequency of line current S            |
| fx3_S      |  The 3rd multiple of the fundamental frequency of line current S             |
| fx5_S      |   The 5th multiple of the fundamental frequency of line current S            |
| fx7_S      |    The 7th multiple of the fundamental frequency of line current S   |
| fx1_T      | The fundamental frequency of line current R         |
| fx0d5_T    | The 0.5th multiple of the fundamental frequency  of line current T  |
| fx1d5_T    | The 1.5th multiple of the fundamental frequency  of line current T             |
| fx2d5_T    |   The 2.5th multiple of the fundamental frequency of line current T            |
| fx3_T      |  The 3rd multiple of the fundamental frequency of line current T             |
| fx5_T      |   The 5th multiple of the fundamental frequency of line current T            |
| fx7_T      |    The 7th multiple of the fundamental frequency of line current T   |


The unique features in [`v000_SCIG_SC_SENSORC_HOS.csv`](v000_SCIG_SC_SENSORC_HOS.csv) are:


|    Field   | Description |
|:----------:|:-----------:|
| Skewness_R      | The Skewness value of of line current R         |
| Kurtosis_R    |The Kurtosis value of of line current R           |
| Variance_R    | The Variance value of of line current R                      |
| RMS_R    |   The RMS value of of line current R                   |
| Skewness_S      | The Skewness value of of line current S         |
| Kurtosis_S    |The Kurtosis value of of line current S           |
| Variance_S    | The Variance value of of line current S                      |
| RMS_S    |   The RMS value of of line current S                   |
| Skewness_T      | The Skewness value of of line current T         |
| Kurtosis_T    |The Kurtosis value of of line current T           |
| Variance_T    | The Variance value of of line current T                      |
| RMS_T    |   The RMS value of of line current T            |