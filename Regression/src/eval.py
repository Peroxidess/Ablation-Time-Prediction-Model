from model.history_ import plot_metric_df
import pandas as pd
import matplotlib.pyplot as plt
import os


xx = os.getcwd()
path_root = '../report/result/'
task_name = 'ablation_time_all'
metric_list = []
metric_list_dir = ['metric_ablation_time_enh_10nrun_1Fold.csv',
'metric_ablation_time_vanilla_10nrun_1Fold.csv',
'metric_gbm_10nrun_1Fold.csv',
'metric_lr_10nrun_1Fold.csv',
]
for metric_dir in metric_list_dir:
    dir = path_root + metric_dir
    metric_df = pd.read_csv(dir)
    metric_list.append(metric_df)
plot_metric_df(metric_list, task_name, val_flag='val_')
plt.show()
pass
