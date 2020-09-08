import pickle 
path = './log_data/wfg2_Saf_init10_MultiSurrogate_mean/wfg2_Saf_init10_MultiSurrogate_meanseed_7270__978ac128-f1c5-11ea-b0f8-40b034171640.pkl'

result = pickle.load(open(path,  "rb"))

print(result.keys())
