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

# tune the model at first session with adapter, and then conduct simplecil.
num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._network = TSVDNet(args, True)

        self.batch_size = args["batch_size"]
        self.tsvd_batch_size = args["tsvd_batch_size"]

        self.init_lr = args["init_lr"]

        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

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

        if self._cur_task == 0:
            self._network.set_phase(1)
            # adapter training
            self._init_train(train_loader, test_loader)

        # clear spaces after parameter-efficient tuning

        for param in self._network.parameters():
            param.grad = None
        self._network.freeze()
        self._network.eval()
        torch.cuda.empty_cache()

        # tsvd
        self._network.set_phase(2)
        with torch.no_grad():
            for i, batch in enumerate(train_loader_for_tsvd):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)

                Features_h = self._network.extract_vector(data)

                self._network.learn_batch(Features_h, label)

        self._network.update_svd()


    def _init_train(self, train_loader, test_loader):
        # show total parameters and trainable parameters
        total_params = sum(p.numel() for p in self._network.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self._network.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        if total_params != total_trainable_params:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,
                                  weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        else:
            assert 0

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'],
                                                         eta_min=self.min_lr)

        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)