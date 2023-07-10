import os

cfg = {
    'dataset_root': './Food-11',
    'save_dir': './outputs',
    'exp_name': "normal1",
    'batch_size': 64,
    'lr': 3e-4,
    'seed': 20220013,
    'loss_fn_type': 'KD', # simple baseline: CE, medium baseline: KD. See the Knowledge_Distillation part for more information.
    'weight_decay': 1e-5,
    'grad_norm_max': 10,
    'n_epochs': 1000, # train more steps to pass the medium baseline.
    'patience': 300,
}

save_path = os.path.join(cfg['save_dir'], cfg['exp_name']) # create saving directory
