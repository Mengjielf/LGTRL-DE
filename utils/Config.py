# -*- coding: utf-8 -*-
class mimicConfig():
    def __init__(self):
        self.seed = 1234
        self.seeds=[1,12,123,1234,12345]

        # data dir
        self.data_path = "./data/processed_data/5fold/"


        # model dir
        self.model_dir="./5_fold_results/mimic_models/"
        self.los_model_dir="./5_fold_results/mimic_los_models/"
        self.history_dir="./5_fold_results/mimic_history/"
        self.los_history_dir="./5_fold_results/mimic_los_history/"
        self.model_variant_dir='./5_fold_results/mimic_variant_models/'
        self.history_variant_dir='./5_fold_results/mimic_variant_history/'

        # task details
        self.k_fold = 5

        #model params
        self.save_dir = 'results/'
        self.embedding_dim = 5
        self.epochs = 100
        self.batch_size = 256

