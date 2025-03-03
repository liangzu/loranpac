import copy
import logging
import torch
from torch import nn
import torch.nn.functional as F

import hickle as hkl

import tensorly as tl
tl.set_backend('pytorch') # or any other backend

import numpy as np
import os

from math import sqrt

from utils.toolkit import truncated_svd, target2onehot

from backbone.linears import SimpleLinear, SplitCosineLinear, CosineLinear, EaseCosineLinear, NormalLinear
from backbone.prompt import CodaPrompt
import timm

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    # SimpleCIL or SimpleCIL w/ Finetune
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()

    elif '_memo' in name:
        if args["model_name"] == "memo":
            from backbone import vit_memo
            _basenet, _adaptive_net = timm.create_model("vit_base_patch16_224_memo", pretrained=True, num_classes=0)
            _basenet.out_dim = 768
            _adaptive_net.out_dim = 768
            return _basenet, _adaptive_net
    # SSF
    elif '_ssf' in name:
        if args["model_name"] == "adam_ssf" or args["model_name"] == "ranpac":
            from backbone import vit_ssf
            if name == "pretrained_vit_b16_224_ssf":
                model = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
                model.out_dim = 768
            elif name == "pretrained_vit_b16_224_in21k_ssf":
                model = timm.create_model("vit_base_patch16_224_in21k_ssf", pretrained=True, num_classes=0)
                model.out_dim = 768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    # VPT
    elif '_vpt' in name:
        if args["model_name"] == "adam_vpt" or args["model_name"] == "ranpac":
            from backbone.vpt import build_promptmodel
            if name == "pretrained_vit_b16_224_vpt":
                basicmodelname = "vit_base_patch16_224"
            elif name == "pretrained_vit_b16_224_in21k_vpt":
                basicmodelname = "vit_base_patch16_224_in21k"

            print("modelname,", name, "basicmodelname", basicmodelname)
            VPT_type = "Deep"
            if args["vpt_type"] == 'shallow':
                VPT_type = "Shallow"
            Prompt_Token_num = args["prompt_token_num"]

            model = build_promptmodel(modelname=basicmodelname, Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type)
            prompt_state_dict = model.obtain_prompt()
            model.load_prompt(prompt_state_dict)
            model.out_dim = 768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    elif '_adapter' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "adam_adapter" or args["model_name"] == "ranpac" or "tsvd" in args["model_name"]:
            from backbone import vit_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
            )
            if name == "pretrained_vit_b16_224_adapter":
                model = vit_adapter.vit_base_patch16_224_adapter(num_classes=0,
                                                                 global_pool=False, drop_path_rate=0.0,
                                                                 tuning_config=tuning_config)
                model.out_dim = 768
            elif name == "pretrained_vit_b16_224_in21k_adapter":
                model = vit_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                                                                       global_pool=False,
                                                                       drop_path_rate=0.0,
                                                                       tuning_config=tuning_config)
                model.out_dim = 768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # L2P
    elif '_l2p' in name:
        if args["model_name"] == "l2p":
            from backbone import vit_l2p
            model = timm.create_model(
                args["backbone_type"],
                pretrained=args["pretrained"],
                num_classes=args["nb_classes"],
                drop_rate=args["drop"],
                drop_path_rate=args["drop_path"],
                drop_block_rate=None,
                prompt_length=args["length"],
                embedding_key=args["embedding_key"],
                prompt_init=args["prompt_key_init"],
                prompt_pool=args["prompt_pool"],
                prompt_key=args["prompt_key"],
                pool_size=args["size"],
                top_k=args["top_k"],
                batchwise_prompt=args["batchwise_prompt"],
                prompt_key_init=args["prompt_key_init"],
                head_type=args["head_type"],
                use_prompt_mask=args["use_prompt_mask"],
            )
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # dualprompt
    elif '_dualprompt' in name:
        if args["model_name"] == "dualprompt":
            from backbone import vit_dualprompt
            model = timm.create_model(
                args["backbone_type"],
                pretrained=args["pretrained"],
                num_classes=args["nb_classes"],
                drop_rate=args["drop"],
                drop_path_rate=args["drop_path"],
                drop_block_rate=None,
                prompt_length=args["length"],
                embedding_key=args["embedding_key"],
                prompt_init=args["prompt_key_init"],
                prompt_pool=args["prompt_pool"],
                prompt_key=args["prompt_key"],
                pool_size=args["size"],
                top_k=args["top_k"],
                batchwise_prompt=args["batchwise_prompt"],
                prompt_key_init=args["prompt_key_init"],
                head_type=args["head_type"],
                use_prompt_mask=args["use_prompt_mask"],
                use_g_prompt=args["use_g_prompt"],
                g_prompt_length=args["g_prompt_length"],
                g_prompt_layer_idx=args["g_prompt_layer_idx"],
                use_prefix_tune_for_g_prompt=args["use_prefix_tune_for_g_prompt"],
                use_e_prompt=args["use_e_prompt"],
                e_prompt_layer_idx=args["e_prompt_layer_idx"],
                use_prefix_tune_for_e_prompt=args["use_prefix_tune_for_e_prompt"],
                same_key_value=args["same_key_value"],
            )
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # Coda_Prompt
    elif '_coda_prompt' in name:
        if args["model_name"] == "coda_prompt":
            from backbone import vit_coda_promtpt
            # model = timm.create_model(args["backbone_type"], pretrained=args["pretrained"])
            model = vit_coda_promtpt.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                            num_heads=12, ckpt_layer=0,
                            drop_path_rate=0)
            from timm.models import vit_base_patch16_224
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            del load_dict['head.weight']; del load_dict['head.bias']
            model.load_state_dict(load_dict)

            return model
    elif '_ease' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "ease":
            from backbone import vit_ease
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
                _device=args["device"][0]
            )
            if name == "vit_base_patch16_224_ease":
                model = vit_ease.vit_base_patch16_224_ease(num_classes=0,
                                                           global_pool=False, drop_path_rate=0.0,
                                                           tuning_config=tuning_config)
                model.out_dim = 768
            elif name == "vit_base_patch16_224_in21k_ease":
                model = vit_ease.vit_base_patch16_224_in21k_ease(num_classes=0,
                                                                 global_pool=False, drop_path_rate=0.0,
                                                                 tuning_config=tuning_config)
                model.out_dim = 768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x["features"])
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.backbone.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.backbone.last_conv.register_forward_hook(
            forward_hook
        )


class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc


class DERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(DERNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.backbones = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.backbones)

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)

        out = self.fc(features)  # {logits: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim:])["logits"]

        out.update({"aux_logits": aux_logits, "features": features})
        return out
        """
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        """

    def update_fc(self, nb_classes):
        if len(self.backbones) == 0:
            self.backbones.append(get_backbone(self.args, self.pretrained))
        else:
            self.backbones.append(get_backbone(self.args, self.pretrained))
            self.backbones[-1].load_state_dict(self.backbones[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = self.backbones[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_backbone(self):
        for param in self.backbones.parameters():
            param.requires_grad = False
        self.backbones.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.backbones) == 1
        self.backbones[0].load_state_dict(model_infos['backbone'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

# class SVDLayer(nn.Module):
#     # maintain a matrix inverse of size ExE (E: embedding dimension)
#     def __init__(self, args):
#         # super().__init__(args, pretrained)
#         super(SVDLayer, self).__init__()
#
#         self.args = args
#         self.num_total_seen_classes = 0
#
#         self._device = args["device"][0]
#
#         # singular vectors and singular values
#         self.U = None
#         self.Sigma = None
#
#         self.Features_h = []
#
#         # self.args["nb_classes"] is the total number of classes. Many implementations use this infomation. The proposed method does not necessarily need it
#         self.cov_HY = torch.zeros(args['E'], self.args["nb_classes"]).to(self._device)
#
#     def update_num_seen_classes(self, num_total_seen_classes):
#         self.num_total_seen_classes = num_total_seen_classes
#
#     def learn_batch(self, Features_h, label):
#         self.cov_HY.index_add_(dim=1, index=label, source=Features_h.T)
#
#         self.Features_h.append(Features_h)
#         # sigma2 = torch.pow(self.Sigma, 2)
#         # stable_rank = torch.sum(sigma2) / torch.max(sigma2)
#         # print(torch.max(sigma2), torch.min(sigma2), torch.div(torch.max(sigma2), torch.min(sigma2)))
#         # print(f"stable rank: {stable_rank}")
#
#     def get_spectrum(self):
#         spectrum = {}
#         sigma2 = torch.pow(self.Sigma, 2)
#
#         spectrum['cond_num'] = torch.max(sigma2).item() / torch.min(sigma2).item()
#         spectrum['stable_rank'] = torch.sum(sigma2).item() / torch.max(sigma2).item()
#         spectrum['sv'] = self.Sigma          # singular values
#         spectrum['ev'] = sigma2              # eigenvalues
#
#         spectrum['min_sv'] = torch.min(self.Sigma).item()
#         spectrum['max_sv'] = torch.max(self.Sigma).item()
#
#         spectrum['sv_L1'] = torch.sum(self.Sigma).item()
#         spectrum['sv_L2'] = torch.sqrt(torch.sum(sigma2)).item()
#
#         return spectrum
#
#     def get_weight(self):
#         # updating singular vectors and singular values
#         self.Features_h = torch.cat(self.Features_h, dim=0)
#
#         if self.U is None:
#             self.U, self.Sigma, _ = torch.linalg.svd(self.Features_h.T, full_matrices=False)
#         else:
#             upper_off_diag = self.U.T @ self.Features_h.T
#
#             Q, R = torch.linalg.qr(self.Features_h.T - self.U @ upper_off_diag)
#
#             lower = torch.cat([torch.zeros(R.shape[0], self.Sigma.shape[0]).to(self._device), R], dim=1)
#
#             upper = torch.cat([torch.diag(self.Sigma), upper_off_diag], dim=1)
#
#             U, self.Sigma, _ = torch.linalg.svd(torch.cat([upper, lower], dim=0), full_matrices=False)
#
#             del R, upper, lower, upper_off_diag
#
#             torch.cuda.empty_cache()
#             # cut near-zero singular vectors and singular values
#             if self.args['rank'] < self.Sigma.shape[0]:
#                 self.Sigma = self.Sigma[:self.args['rank']]
#                 U = U[:, :self.args['rank']]
#
#             torch.cuda.empty_cache()
#             self.U = torch.cat([self.U, Q], dim=1)
#             self.U, _ = torch.linalg.qr(self.U @ U)
#
#         self.Features_h = []  # the information of self.Features_h has now been stored in self.U and self.Sigma
#
#         if self.args['ridge_type'] == 'l1':
#             ridge = torch.sum(self.Sigma).item()
#         elif self.args['ridge_type'] == 'l2':
#             ridge = torch.linalg.vector_norm(self.Sigma, ord=2).item()
#         else:
#             assert False
#
#         weight = self.U.T @ self.cov_HY[:, :self.num_total_seen_classes]
#         weight = torch.div(weight.T, torch.pow(self.Sigma, 2) + ridge).T
#
#         return (self.U @ weight).T
#
#     def forward(self, x):
#         if self.args['coslinear']:
#             weight = self.U.T @ self.cov_HY[:, :self.num_total_seen_classes]
#             weight = torch.div(weight.T, torch.pow(self.Sigma, 2) + self.ridge).T
#
#             weight = self.U @ weight
#             out = F.normalize(x, p=2, dim=1) @ F.normalize(weight, p=2, dim=0)
#
#         else:
#             out = x @ self.U
#             out = torch.div(out, torch.pow(self.Sigma, 2) + self.ridge) @ self.U.T
#
#             out = out @ self.cov_HY[:, :self.num_total_seen_classes]
#
#         return {"logits": out, "features": x}
#
#
# class WoodburyLayer(nn.Module):
#     # maintain a matrix inverse of size ExE (E: embedding dimension)
#     def __init__(self, args):
#         # super().__init__(args, pretrained)
#         super(WoodburyLayer, self).__init__()
#
#         self.args = args
#         self.num_total_seen_classes = 0
#
#         self._device = args["device"][0]
#
#         ridge = 10 * args['E']
#         print('ridge', ridge)
#         self.G_inv = torch.eye(args['E']).to(self._device) / ridge
#
#         # self.args["nb_classes"] is the total number of classes. Many implementations use this infomation. The proposed method does not necessarily need it
#         self.cov_HY = torch.zeros(args['E'], self.args["nb_classes"]).to(self._device)
#
#     def update_num_seen_classes(self, num_total_seen_classes):
#         self.num_total_seen_classes = num_total_seen_classes
#
#     def learn_batch(self, Features_h, label):
#         batch_size = Features_h.size(dim=0)
#
#         self.cov_HY.index_add_(dim=1, index=label, source=Features_h.T)
#
#         # Below we implement the Woodbury matrix identity (https://en.wikipedia.org/wiki/Woodbury_matrix_identity)
#         tmp1 = self.G_inv @ Features_h.T
#
#         id_batch = torch.eye(batch_size).to(self._device)  # this depends on batch size
#
#         tmp2 = torch.linalg.inv(id_batch + Features_h @ tmp1)  # invert a small matrix of size (batch_size x batch_size)
#
#         self.G_inv -= tmp1 @ tmp2 @ tmp1.T
#
#     def get_weight(self):
#         return (self.G_inv @ self.cov_HY[:, :self.num_total_seen_classes]).T
#     def forward(self, x):
#         if self.args['coslinear']:
#             weight = self.G_inv @ self.cov_HY[:, :self.num_total_seen_classes]
#             out = F.normalize(x, p=2, dim=1) @ F.normalize(weight, p=2, dim=0)
#         else:
#             out = (x @ self.G_inv) @ self.cov_HY[:, :self.num_total_seen_classes]
#
#         return {"logits": out, "features": x}


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        # for RanPAC
        self.W_rand = None
        self.RP_dim = None

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        if self.RP_dim is not None:
            feature_dim = self.RP_dim
        else:
            feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        if self.W_rand is not None:
            x = torch.nn.functional.relu(x @ self.W_rand)
        out = self.fc(x)
        out.update({"features": x})
        return out




class TSVDNet(BaseNet):
    # maintain a low-rank SVD throughout learning
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

        self.args = args
        self.num_total_seen_classes = 0

        # random projection, as in RanPAC
        self.RE_dim = None
        self.RE = None


        if self.args['use_RE']:
            self.init_RE()
        else:
            self.dim_pre_classifier = self.feature_dim

        self.ridge = self.args['ridge']

        self.Features_h = []

        self.fc = None

        self.phase = -1
        # singular vectors and singular values
        self.U = None
        self.Sigma = None
        self.Sigma_all = None

        self.num_total_samples = 0
        self.num_current_samples = 0

        self.spectrums = []

        # self.args["nb_classes"] is the total number of classes. Many implementations use this infomation. The proposed method does not necessarily need it
        self.cov_HY = torch.zeros(self.dim_pre_classifier, self.args["nb_classes"]).to(self._device)

        self.max_rank = self.args['rank']
        r = round(self.dim_pre_classifier * (1 - self.args['truncate_percent'] / 100))
        self.max_rank = min(r, self.max_rank)

        if self.args['save_spectrum']:
            self.max_truncated_evs = []
            self.save_spectrum_count = 0

    def init_RE(self):
        self.RE_dim = self.args['E']
        self.dim_pre_classifier = self.RE_dim

        self.RE = torch.randn(self.feature_dim, self.RE_dim).to(self._device)


    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.dim_pre_classifier, nb_classes).to(self._device)

        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.dim_pre_classifier).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        # fc = SimpleLinear(in_dim, out_dim, bias=False)

        return fc

    def extract_vector(self, x):
        return self.backbone(x)

    def set_phase(self, phase):
        self.phase = phase

    def get_spectrum(self):
        spectrum = {}
        sigma2 = torch.pow(self.Sigma, 2)

        spectrum['cond_num'] = torch.max(sigma2).cpu().item() / torch.min(sigma2).cpu().item()
        spectrum['stable_rank'] = torch.sum(sigma2).cpu().item() / torch.max(sigma2).cpu().item()
        spectrum['sv'] = self.Sigma.cpu().numpy()          # singular values
        spectrum['ev'] = sigma2.cpu().numpy()              # eigenvalues

        spectrum['min_sv'] = torch.min(self.Sigma).cpu().item()
        spectrum['max_sv'] = torch.max(self.Sigma).cpu().item()

        spectrum['sv_L1'] = torch.sum(self.Sigma).cpu().item()
        spectrum['sv_L2'] = torch.sqrt(torch.sum(sigma2)).cpu().item()

        spectrum['max_truncated_evs'] = self.max_truncated_evs

        if self.args['truncate_percent'] > 0 and self.Sigma_all is not None:
            spectrum['ev_all'] = torch.pow(self.Sigma_all, 2).cpu().numpy()

        return spectrum

    def get_residual(self):
        pass

    def update_num_seen_classes(self, num_total_seen_classes):
        self.num_total_seen_classes = num_total_seen_classes

    def learn_batch(self, Features_h, label):
        self.num_total_samples += Features_h.size(0)
        self.num_current_samples += Features_h.size(0)

        if self.args['use_RE']:
            Features_h = Features_h @ self.RE

        if self.args['use_relu']:
            Features_h = F.relu(Features_h)

        self.cov_HY.index_add_(dim=1, index=label, source=Features_h.T)

        self.Features_h.append(Features_h)

        current_batch_size = len(self.Features_h) * Features_h.size(0)
        if current_batch_size > 10000:
            self.update_svd()

        # if self.U is not None:
        #     # we need to make sure the number of preserved singular values is smaller than embedding dimension
        #     # dim_pre_classifier
        #     num = self.U.size(1) + current_batch_size
        #
        #     if num < self.dim_pre_classifier and num + Features_h.size(0) >= self.dim_pre_classifier:
        #         self.update_svd()

    def update_svd(self):
        if len(self.Features_h) == 0:
            return

        self.Features_h = torch.cat(self.Features_h, dim=0)

        num_preserved = round(self.num_total_samples * (1 - self.args['truncate_percent'] / 100))
        num_preserved = min(num_preserved, self.max_rank)

        # updating singular vectors and singular values
        if self.U is None:
            # print(self.Features_h.size())
            self.U, self.Sigma, _ = torch.linalg.svd(self.Features_h.T, full_matrices=False)

            if self.args['save_spectrum'] and self.args['truncate_percent'] > 0:
                self.Sigma_all = self.Sigma
                self.max_truncated_evs.append(self.Sigma[num_preserved].cpu().item())

            self.U = self.U[:, :num_preserved]
            self.Sigma = self.Sigma[:num_preserved]

            self.Features_h = []
            self.num_current_samples = 0
        else:
            upper_off_diag = self.U.T @ self.Features_h.T

            Q, R = torch.linalg.qr(self.Features_h.T - self.U @ upper_off_diag)

            self.Features_h = []
            self.num_current_samples = 0

            lower = torch.cat([torch.zeros(R.shape[0], self.Sigma.shape[0]).to(self._device), R], dim=1)

            upper = torch.cat([torch.diag(self.Sigma), upper_off_diag], dim=1)

            U, self.Sigma, _ = torch.linalg.svd(torch.cat([upper, lower], dim=0), full_matrices=False)

            del R, upper, lower, upper_off_diag

            torch.cuda.empty_cache()

            if self.args['save_spectrum'] and self.args['truncate_percent'] > 0:
                self.Sigma_all = self.Sigma
                self.max_truncated_evs.append(self.Sigma[num_preserved].cpu().item())

            U = U[:, :num_preserved]
            self.Sigma = self.Sigma[:num_preserved]

            # if self.max_rank < self.Sigma.shape[0]:
            #     self.Sigma = self.Sigma[:self.max_rank]
            #     U = U[:, :self.max_rank]

            self.U = torch.cat([self.U, Q], dim=1) @ U
            self.U, _ = torch.linalg.qr(self.U)

        torch.cuda.empty_cache()

        if self.args['save_spectrum']:
            self.spectrums.append(self.get_spectrum())
            self.save_spectrum()

    def save_spectrum(self):
        dataset_name = self.args['dataset']
        backbone_name = self.args['backbone_type']
        model_name = self.args['model_name']
        E = self.args['E']

        root = './results'
        if not os.path.exists(root):
            os.makedirs(root)

        if self.args['truncate_percent'] > 0:
            fn = f'{root}/E={E}-continual-{model_name}-{backbone_name}-{dataset_name}.hkl'

            if self.args['use_RE'] and self.args['use_relu']:
                fn = f'{root}/spectrum-continual-E={E}-RE-relu-{model_name}-{backbone_name}-{dataset_name}.hkl'

            if self.args['use_RE'] and not self.args['use_relu']:
                fn = f'{root}/spectrum-continual-E={E}-RE-{model_name}-{backbone_name}-{dataset_name}.hkl'

            if not self.args['use_RE'] and self.args['use_relu']:
                fn = f'{root}/spectrum-continual-relu-{model_name}-{backbone_name}-{dataset_name}.hkl'

            if not self.args['use_RE'] and not self.args['use_relu']:
                fn = f'{root}/spectrum-continual-{model_name}-{backbone_name}-{dataset_name}.hkl'
        else:
            if self.args['use_RE'] and self.args['use_relu']:
                fn = f'{root}/spectrum-offline-E={E}-RE-relu-{model_name}-{backbone_name}-{dataset_name}.hkl'

            if self.args['use_RE'] and not self.args['use_relu']:
                fn = f'{root}/spectrum-offline-E={E}-RE-{model_name}-{backbone_name}-{dataset_name}.hkl'

            if not self.args['use_RE'] and self.args['use_relu']:
                fn = f'{root}/spectrum-offline-relu-{model_name}-{backbone_name}-{dataset_name}.hkl'

            if not self.args['use_RE'] and not self.args['use_relu']:
                fn = f'{root}/spectrum-offline-{model_name}-{backbone_name}-{dataset_name}.hkl'

        hkl.dump(self.spectrums, fn)

        # np.save(fn, self.spectrums)

    def forward(self, x):
        x = self.backbone(x) # batch size x d

        # x_norm = torch.pow(torch.norm(x, dim=1), 2)

        if self.args['use_RE']:
            x = x @ self.RE # batch size x E

        # re_norm = torch.pow(torch.norm(self.RE, dim=0), 2)

        if self.args['use_relu']:
            x = F.relu(x)

        if self.phase == 1:
            out = self.fc(x)
            out.update({"features": x})
            return out

        if self.phase == 2:
            U = self.U
            Sigma = self.Sigma

            Ut_cov_HY = U.T @ self.cov_HY[:, :self.num_total_seen_classes]

            weight_ridge = U @ torch.div(Ut_cov_HY.T, torch.pow(Sigma, 2) + self.ridge).T # rank x num_class

            if self.args['coslinear']:
                out = F.normalize(x, p=2, dim=1) @ F.normalize(weight_ridge, p=2, dim=0)  # batch_size x num_class
            else:
                out = x @ weight_ridge

            return {"logits": out, "features": x}

        assert False




# l2p and dualprompt
class PromptVitNet(nn.Module):
    def __init__(self, args, pretrained):
        super(PromptVitNet, self).__init__()
        self.backbone = get_backbone(args, pretrained)
        if args["get_original_backbone"]:
            self.original_backbone = self.get_original_backbone(args)
        else:
            self.original_backbone = None

    def get_original_backbone(self, args):
        return timm.create_model(
            args["backbone_type"],
            pretrained=args["pretrained"],
            num_classes=args["nb_classes"],
            drop_rate=args["drop"],
            drop_path_rate=args["drop_path"],
            drop_block_rate=None,
        ).eval()

    def forward(self, x, task_id=-1, train=False):
        with torch.no_grad():
            if self.original_backbone is not None:
                cls_features = self.original_backbone(x)['pre_logits']
            else:
                cls_features = None

        x = self.backbone(x, task_id=task_id, cls_features=cls_features, train=train)
        return x


# coda_prompt
class CodaPromptVitNet(nn.Module):
    def __init__(self, args, pretrained):
        super(CodaPromptVitNet, self).__init__()
        self.args = args
        self.backbone = get_backbone(args, pretrained)
        self.fc = nn.Linear(768, args["nb_classes"])
        self.prompt = CodaPrompt(768, args["nb_tasks"], args["prompt_param"])

    # pen: get penultimate features
    def forward(self, x, pen=False, train=False):
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.backbone(x)
                q = q[:, 0, :]
            out, prompt_loss = self.backbone(x, prompt=self.prompt, q=q, train=train)
            out = out[:, 0, :]
        else:
            out, _ = self.backbone(x)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.fc(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out


class MultiBranchCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

        # no need the backbone.

        print(
            'Clear the backbone in MultiBranchCosineIncrementalNet, since we are using self.backbones with dual branches')
        self.backbone = torch.nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbones = nn.ModuleList()
        self.args = args

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self._feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self._feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]

        features = torch.cat(features, 1)
        # import pdb; pdb.set_trace()
        out = self.fc(features)
        out.update({"features": features})
        return out

    def construct_dual_branch_network(self, tuned_model):
        if 'ssf' in self.args['backbone_type']:
            newargs = copy.deepcopy(self.args)
            newargs['backbone_type'] = newargs['backbone_type'].replace('_ssf', '')
            print(newargs['backbone_type'])
            self.backbones.append(get_backbone(newargs))  # pretrained model without scale
        elif 'vpt' in self.args['backbone_type']:
            newargs = copy.deepcopy(self.args)
            newargs['backbone_type'] = newargs['backbone_type'].replace('_vpt', '')
            print(newargs['backbone_type'])
            self.backbones.append(get_backbone(newargs))  # pretrained model without vpt
        elif 'adapter' in self.args['backbone_type']:
            newargs = copy.deepcopy(self.args)
            newargs['backbone_type'] = newargs['backbone_type'].replace('_adapter', '')
            print(newargs['backbone_type'])
            self.backbones.append(get_backbone(newargs))  # pretrained model without adapter
        else:
            self.backbones.append(get_backbone(self.args))  # the pretrained model itself

        self.backbones.append(tuned_model.backbone)  # adappted tuned model

        self._feature_dim = self.backbones[0].out_dim * len(self.backbones)
        self.fc = self.generate_fc(self._feature_dim, self.args['init_cls'])


class FOSTERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(FOSTERNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.backbones = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None
        self.args = args

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.backbones)

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim:])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        self.backbones.append(get_backbone(self.args, self.pretrained))
        if self.out_dim is None:
            self.out_dim = self.backbones[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.backbones[-1].load_state_dict(self.backbones[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_backbone(self):
        for param in self.backbones.parameters():
            param.requires_grad = False
        self.backbones.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format(
                args["dataset"],
                args["seed"],
                args["backbone_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.backbones) == 1
        self.backbones[0].load_state_dict(model_infos['backbone'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class AdaptiveNet(nn.Module):
    def __init__(self, args, pretrained):
        super(AdaptiveNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.TaskAgnosticExtractor, _ = get_backbone(args, pretrained)  # Generalized blocks
        self.TaskAgnosticExtractor.train()
        self.AdaptiveExtractors = nn.ModuleList()  # Specialized Blocks
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.AdaptiveExtractors)

    def extract_vector(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        out = self.fc(features)  # {logits: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim:])["logits"]

        out.update({"aux_logits": aux_logits, "features": features})
        out.update({"base_features": base_feature_map})
        return out

        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''

    def update_fc(self, nb_classes):
        _, _new_extractor = get_backbone(self.args, self.pretrained)
        if len(self.AdaptiveExtractors) == 0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
            self.AdaptiveExtractors[-1].load_state_dict(self.AdaptiveExtractors[-2].state_dict())

        if self.out_dim is None:
            # logging.info(self.AdaptiveExtractors[-1])
            self.out_dim = self.AdaptiveExtractors[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, :self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = (torch.norm(weights[-increment:, :], p=2, dim=1))
        oldnorm = (torch.norm(weights[:-increment, :], p=2, dim=1))
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print('alignweights,gamma=', gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format(
                args["dataset"],
                args["seed"],
                args["backbone_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        checkpoint_name = checkpoint_name.replace("memo_", "")
        model_infos = torch.load(checkpoint_name)
        model_dict = model_infos['backbone']
        assert len(self.AdaptiveExtractors) == 1

        base_state_dict = self.TaskAgnosticExtractor.state_dict()
        adap_state_dict = self.AdaptiveExtractors[0].state_dict()

        pretrained_base_dict = {
            k: v
            for k, v in model_dict.items()
            if k in base_state_dict
        }

        pretrained_adap_dict = {
            k: v
            for k, v in model_dict.items()
            if k in adap_state_dict
        }

        base_state_dict.update(pretrained_base_dict)
        adap_state_dict.update(pretrained_adap_dict)

        self.TaskAgnosticExtractor.load_state_dict(base_state_dict)
        self.AdaptiveExtractors[0].load_state_dict(adap_state_dict)
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class EaseNet(BaseNet):
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim = self.backbone.out_dim
        self.fc = None
        self.use_init_ptm = args["use_init_ptm"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]

    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            # print(name)

    @property
    def feature_dim(self):
        if self.use_init_ptm:
            return self.out_dim * (self._cur_task + 2)
        else:
            return self.out_dim * (self._cur_task + 1)

    # (proxy_fc = cls * dim)
    def update_fc(self, nb_classes):
        self._cur_task += 1

        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            self.proxy_fc = self.generate_fc(self.out_dim, self.inc).to(self._device)

        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()

        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            fc.weight.data[: old_nb_classes, : -self.out_dim] = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = EaseCosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, test=False):
        if test == False:
            x = self.backbone.forward(x, False)
            out = self.proxy_fc(x)
        else:
            x = self.backbone.forward(x, True, use_init_ptm=self.use_init_ptm)
            if self.args["moni_adam"] or (not self.args["use_reweight"]):
                out = self.fc(x)
            else:
                out = self.fc.forward_reweight(x, cur_task=self._cur_task, alpha=self.alpha, init_cls=self.init_cls,
                                               inc=self.inc, use_init_ptm=self.use_init_ptm, beta=self.beta)

        out.update({"features": x})
        return out

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())