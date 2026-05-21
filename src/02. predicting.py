## initialize torch and CUDA
# import torch
# torch.cuda.is_available()
# from main import main

# # %% PREDICTION 
# main(["-mo", "predict"
#       , "-pcf", "data.yaml"
#       ]) 
# #!python main.py -mo=predict -pcf=data.yaml

##
from pipeline_predict import PipeLinePredict
pp = PipeLinePredict("data.yaml")
# using the following data.yaml:
# model_info:
#     folder: MODEL_fileA__NT_fileB__NT_ONE_DATASET_2026-05-20_subphr_100ep
#     model_config: mdlCfgseq2seq_len10_lr0.0003_ep100_emb256_h4_l2.json
#     model: seq2seq_len10_lr0.0003_ep100_emb256_h4_l2.pth
# new_data: ../sp_data/fileA__NT
# output: fileB__NT_predicted
# # output: C:/temp/fileB__NT_predicted
# predict_idx: 0
# beam_size: 1
# beam_alpha: 0




