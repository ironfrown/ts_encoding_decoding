# Time series encoding and decoding
Here is a temporary repository for QTML poster / future journal paper.

## Folders
- log_2: repository of files generated by notebooks
- utils: various utilities, many obsolete
- runs: archive of important runs from the past 
- legacy: obsolete files

## File ready to use
- **<font color="red">ts_qiskit_versions_and_issues.ipynb:</font>**<br/>List of versions and unresolved issues/bugs
- **qtsa_00_data_v1.0_2sin.ipynb:**<br/>Generates simple TS for PQFT models
- **qtsa_00_data_v1.0_2sin_sw.ipynb:**<br/>Generates TS for SW models
- **qtsa_02_serial_pqft_vXXX_train/analysis_comment.ipynb:**<br/>Trains and analyses PQFT serial (1 qubit) model
- **qtsa_03_parallel_pqft_vXXX_train/analysis_comment.ipynb:**<br/>Trains and analyses PQFT parallel (multi-qubit) model
- **qtsa_04_xqnn_vXXX_train/analysis_comment.ipynb:**<br/>Trains and analyses an extended SW QNN model
- **qtsa_05_swind_vXXX_train/analysis_comment.ipynb:**<br/>Trains and analyses SW QTSA model
- **qtsa_06_cnn_vXXX_train/analysis_comment.ipynb: Classic MPL:**<br/>Trains and analyses classical NN (MLP)

## How to proceed
A typical process is to:
- Generate data for model training and save it in **log/data/sub-dir** folder,<br/>
  examples "qtsa_00_data_v1.0_2sin.ipynb" and "qtsa_00_data_v1.0_2sin_sw.ipynb"
- Create a model, train using the data from **log/data/sub-dir** folder (point to it as *DATA_ID*),<br/>
  then save the history of cost in **log/training/sub-dir** and parameters in **log/params/sub-dir**,<br/>
  example "qtsa_02_serial_pqft_v8.16_train_lay27_lbfgsb_ep39.ipynb"
- Analyse cost and evolving models (defined by parameters) from **log/training/sub-dir** (point to it as *train_info_id*),<br/>
  then save charts in *log/figures/sub-dir* and analysis results in **log/analysis/sub-dir**,<br/>
  example "qtsa_03_parallel_pqft_v8.16_analysis_q5_lbfgsb_ep50.ipynb"