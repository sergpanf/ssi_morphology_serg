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
#     folder: MODEL_fileA__NT_fileB__NT_ONE_DATASET
#     model_config: mdlCfgseq2seq_len10_lr0.0003_ep40_emb256_h4_l2.json
#     model: seq2seq_len10_lr0.0003_ep40_emb256_h4_l2.pth
# new_data: ../sp_data/fileA__NT
# output: fileB__NT_predicted
# predict_idx: 0
# beam_size: 1 # 3
# beam_alpha: 0 # 0.75


