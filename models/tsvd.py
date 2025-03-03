import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import TSVDNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

import time
import os

# tune the model at first session with adapter, and then conduct simplecil.
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._network = TSVDNet(args, True)
        self._network.set_phase(2)

        self.batch_size = args["batch_size"]
        self.tsvd_batch_size = args["tsvd_batch_size"]

        self.args = args

        self.times = {}
        self.times['feature'] = 0
        self.times['algorithm'] = 0

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_num_seen_classes(self._total_classes)
        self._network.update_fc(self._total_classes)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                              source="train", mode="test", )
        self.train_loader_for_tsvd = DataLoader(train_dataset_for_protonet, batch_size=self.tsvd_batch_size,
                                                shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_tsvd)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_tsvd):
        self._network.to(self._device)

        with torch.no_grad():
            for i, batch in enumerate(train_loader_for_tsvd):
                torch.cuda.synchronize()
                feature_start = time.time()

                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)

                Features_h = self._network.extract_vector(data)

                torch.cuda.synchronize()
                self.times['feature'] += time.time() - feature_start

                algorithm_start = time.time()
                self._network.learn_batch(Features_h, label)
                self.times['algorithm'] += time.time() - algorithm_start

            algorithm_start = time.time()
            self._network.update_svd()

            torch.cuda.synchronize()
            self.times['algorithm'] += time.time() - algorithm_start