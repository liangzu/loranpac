import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils.toolkit import target2onehot
from torch.nn import functional as F

num_workers = 8


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)

def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    # data_manager = DataManager(
    #     args["dataset"],
    #     args["shuffle"],
    #     args["seed"],
    #     args["init_cls"],
    #     args["increment"],
    #     args,
    # )

    if args['dataset'] == 'core50':
        dil_tasks = ['s1', 's2', 's4', 's5', 's6', 's8', 's9', 's11']
        num_tasks = len(dil_tasks)
        is_dil = True
    elif args['dataset'] == 'cddb':
        dil_tasks = ['gaugan', 'biggan', 'wild', 'whichfaceisreal', 'san']
        num_tasks = len(dil_tasks)
        is_dil = True
    elif args['dataset'] == 'domainnet':
        dil_tasks = ['real', 'quickdraw', 'painting', 'sketch', 'infograph', 'clipart']
        num_tasks = len(dil_tasks)
        is_dil = True
    elif 'joint' in args['dataset']:
        # joint training on DIL datasets
        data_manager = DataManager(
            args["dataset"],
            args["shuffle"],
            args["seed"],
            args["init_cls"],
            args["increment"],
            args,
        )
        assert data_manager.nb_tasks == 1
        num_tasks = data_manager.nb_tasks
        args["nb_tasks"] = num_tasks
        args["nb_classes"] = data_manager.nb_classes

        model = factory.get_model(args["model_name"], args)
        is_dil = False

    else:
        # cil datasets
        is_dil = False
        data_manager = DataManager(
            args["dataset"],
            args["shuffle"],
            args["seed"],
            args["init_cls"],
            args["increment"],
            args,
        )
        num_tasks = data_manager.nb_tasks
        args["nb_tasks"] = num_tasks
        args["nb_classes"] = data_manager.nb_classes

        model = factory.get_model(args["model_name"], args)
        model.is_dil = is_dil

    cnn_curve = {"top1": [], "top5": []}
    cnn_matrix = []

    # for n, p in model._network.named_parameters():
    #     if p.requires_grad:
    #         print("require grad:", n)

    for task in range(num_tasks):
        if is_dil:
            # reset the data manager to the next domain
            print(args["dataset"] + '_' + dil_tasks[task])
            data_manager = DataManager(
                args["dataset"] + '_' + dil_tasks[task],
                args["shuffle"],
                args["seed"],
                args["init_cls"],
                args["increment"],
                args
            )

            args["nb_classes"] = data_manager.nb_classes  # update args

            # initialize model for domain-incremental learning
            if task == 0:
                model = factory.get_model(args["model_name"], args)

            if args['dataset'] == 'cddb':
                model.topk = 2

            model.dil_init = task == 0

            model.is_dil = is_dil

            model._cur_task = -1
            model._known_classes = 0
            model._classes_seen_so_far = 0

        # logging.info("All params: {}".format(count_parameters(model._network)))
        # logging.info(
        #     "Trainable params: {}".format(count_parameters(model._network, True))
        # )
        model.incremental_train(data_manager)

        cnn_accy, nme_accy = model.eval_task()
        model.after_task()


        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
        cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
        cnn_matrix.append(cnn_values)

        cnn_curve["top1"].append(cnn_accy["top1"])
        # cnn_curve["top5"].append(cnn_accy["top5"])

        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        # logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

        print('Average Accuracy (CNN):', sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
        logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))

    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        # forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print('Accuracy Matrix (CNN):')
        print(np_acctable)

        # logging.info('Forgetting (CNN): {}'.format(forgetting))

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))