# LGTRL-DE: Local and Global Temporal Representation Learning with Demographic Embedding for In-hospital Mortality Prediction

The source code for *LGTRL-DE: Local and Global Temporal Representation Learning with Demographic Embedding for In-hospital Mortality Prediction*.

Thanks for your interest in our work.

## Requirements

* We use Python 3.6.6, Keras 2.3.1.
* If you plan to use GPU computation, install CUDA

## Data preparation
We do not provide the MIMIC-III data itself. You must acquire the data yourself from https://mimic.physionet.org/. Specifically, download the CSVs. To run in-hospital mortality prediction task on MIMIC-III bechmark dataset, you should first build benchmark dataset according to https://github.com/YerevaNN/mimic3-benchmarks/.

After building the **in-hospital mortality** dataset, please save the files in ```in-hospital-mortality``` directory to ```/data/row_data/``` directory.

## Run LGTRL-DE

1. Clone the repo.

       git clone https://github.com/Mengjielf/LGTRL-DE/
       cd LGTRL-DE/
    
2. Run the command to obtain processed data. The processed data are saved in /data/processed_data/ file.

       python data_process.py

3. The trained models are saved in /5_fold_results/mimic_models/ file. Run the following command to test model.

       CUDA_VISIBLE_DEVICES=0 python test.py
       


