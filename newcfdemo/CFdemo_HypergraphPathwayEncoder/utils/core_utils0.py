from ast import Lambda
import numpy as np
import pdb
import os
from custom_optims.radam import RAdam
from models.model_ABMIL import ABMIL
from models.model_DeepMISL import DeepMISL
from models.model_MLPOmics import MLPOmics
from models.model_MLPWSI import MLPWSI
# from models.model_SNNOmics0805 import SNNOmics
from models.model_SNNOmics import SNNOmics
from models.model_MaskedOmics import MaskedOmics
from models.model_MCATPathways import MCATPathways
from models.model_SurvPath import SurvPath
from models.model_SurvPath_with_nystrom import SurvPath_with_nystrom
from models.model_TMIL import TMIL
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import pickle

from transformers import (
    get_constant_schedule_with_warmup, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup
)


#----> pytorch imports
import torch
from torch.nn.utils.rnn import pad_sequence

from utils.general_utils import _get_split_loader, _print_network, _save_splits
from utils.loss_func import NLLSurvLoss, MultiTaskLoss

import torch.optim as optim



def _get_splits(datasets, cur, args):
    r"""
    Summarize the train and val splits and return them individually
    
    Args:
        - datasets : tuple
        - cur : Int 
        - args: argspace.Namespace
    
    Return:
        - train_split : SurvivalDataset
        - val_split : SurvivalDataset
    
    """

    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    _save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    return train_split,val_split


def _init_loss_function(args):
    r"""
    Init the survival loss function
    
    Args:
        - args : argspace.Namespace 
    
    Returns:
        - loss_fn : NLLSurvLoss or MultiTaskLoss
    
    """
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'nll_surv':
        if args.enable_multitask:
            loss_fn = MultiTaskLoss(
                survival_alpha=args.alpha_surv,
                multitask_weight=args.multitask_weight
            )
        else:
            # loss_fn = NLLSurvLoss(alpha=args.alpha_surv) # 改
            loss_fn = MultiTaskLoss(
                survival_alpha=args.alpha_surv,
                multitask_weight=0
            )
    else:
        raise NotImplementedError
    print('Done!')
    return loss_fn

def _init_optim(args, model):
    r"""
    Init the optimizer 
    
    Args: 
        - args : argspace.Namespace 
        - model : torch model 
    
    Returns:
        - optimizer : torch optim 
    """
    print('\nInit optimizer ...', end=' ')
    
    # 伪代码示例
    # optimizer = torch.optim.AdamW([
    #     {'params': model.fc_omic.parameters(), 'lr': 1e-4},
    #     {'params': model.survival_classifier.parameters(), 'lr': 1e-4},
    #     {'params': model.clinical_bert_model.parameters(), 'lr': 2e-5} # 为BERT设置一个更小的学习率
    # ], weight_decay=0.01)

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "radam":
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "lamb":
        optimizer = Lambda(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError

    return optimizer

def _init_model(args):
    
    print('\nInit Model...', end=' ')
    if args.type_of_path == "xena":
        omics_input_dim = 1577
    elif args.type_of_path == "hallmarks":
        omics_input_dim = 4241
    elif args.type_of_path == "combine":
        omics_input_dim = 4999
    elif args.type_of_path == "multi":
        if args.study == "tcga_brca":
            omics_input_dim = 9947
        else:
            omics_input_dim = 14933
    else:
        omics_input_dim = 0
    
    # omics baselines
    if args.modality == "mlp_per_path":

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "dropout" : args.encoder_dropout, "num_classes" : args.n_classes,
        }
        model = MaskedOmics(**model_dict)

    elif args.modality == "omics":

        model_dict = {
             "input_dim" : omics_input_dim, "projection_dim": 64, "dropout": args.encoder_dropout
        }
        model = MLPOmics(**model_dict)

    elif args.modality == "snn":

        model_dict = {
             "omic_input_dim" : omics_input_dim,
             "n_stage_classes": getattr(args.dataset_factory, 'n_stage_classes', 4),
             "enable_multitask": args.enable_multitask,
             "multitask_weight": args.multitask_weight
        }
        model = SNNOmics(**model_dict)

    elif args.modality in ["abmil_wsi", "abmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = ABMIL(**model_dict)

    # unimodal and multimodal baselines
    elif args.modality in ["deepmisl_wsi", "deepmisl_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = DeepMISL(**model_dict)

    elif args.modality == "mlp_wsi":
        
        model_dict = {
            "wsi_embedding_dim":args.encoding_dim, "input_dim_omics":omics_input_dim, "dropout":args.encoder_dropout,
            "device": args.device

        }
        model = MLPWSI(**model_dict)

    elif args.modality in ["transmil_wsi", "transmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = TMIL(**model_dict)

    elif args.modality == "coattn":

        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCATPathways(**model_dict)

    elif args.modality == "coattn_motcat":

        model_dict = {
            'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes,
            "ot_reg":0.1, "ot_tau":0.5, "ot_impl":"pot-uot-l2"
        }
        model = MCATPathwaysMotCat(**model_dict)

    # survpath 
    elif args.modality == "survpath":

        model_dict = {'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes}

        if args.use_nystrom:
            model = SurvPath_with_nystrom(**model_dict)
        else:
            model = SurvPath(**model_dict)

    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))

    print('Done!')
    _print_network(args.results_dir, model)

    return model

def _init_loaders(args, train_split, val_split):
    r"""
    Init dataloaders for the train and val datasets 

    Args:
        - args : argspace.Namespace 
        - train_split : SurvivalDataset 
        - val_split : SurvivalDataset 
    
    Returns:
        - train_loader : Pytorch Dataloader 
        - val_loader : Pytorch Dataloader

    """

    print('\nInit Loaders...', end=' ')
    if train_split:
        train_loader = _get_split_loader(args, train_split, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    else:
        train_loader = None

    if val_split:
        val_loader = _get_split_loader(args, val_split,  testing=False, batch_size=1)
    else:
        val_loader = None
    print('Done!')

    return train_loader,val_loader

def _extract_survival_metadata(train_loader, val_loader):
    r"""
    Extract censorship and survival times from the train and val loader and combine to get numbers for the fold
    We need to do this for train and val combined because when evaulating survival metrics, the function needs to know the 
    distirbution of censorhsip and survival times for the trainig data
    
    Args:
        - train_loader : Pytorch Dataloader
        - val_loader : Pytorch Dataloader
    
    Returns:
        - all_survival : np.array
    
    """

    all_censorships = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.censorship_var].to_numpy()],
        axis=0)

    all_event_times = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.label_col].to_numpy()],
        axis=0)

    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    return all_survival

def _unpack_data(modality, device, data, enable_multitask=False):
    r"""
    Depending on the model type, unpack the data and put it on the correct device
    
    Args:
        - modality : String 
        - device : torch.device 
        - data : tuple 
        - enable_multitask : Boolean
    
    Returns:
        - data_WSI : torch.Tensor
        - mask : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - data_omics : torch.Tensor
        - clinical_data_list : list
        - mask : torch.Tensor
        - stage_labels : torch.Tensor (if multitask enabled)
    
    """
    
    stage_labels = None
    
    if modality in ["mlp_per_path", "omics", "snn"]:
        data_WSI = data[0]
        mask = None
        data_omics = data[1].to(device)
        if enable_multitask:
            y_disc, event_time, censor, clinical_data_list, stage_labels = data[2], data[3], data[4], data[5], data[6]
            stage_labels = stage_labels.to(device)
        else:
            y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
    
    elif modality in ["mlp_per_path_wsi", "abmil_wsi", "abmil_wsi_pathways", "deepmisl_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        
        mask_idx = 6 if not enable_multitask else 7
        if data[mask_idx][0,0] == 1:
            mask = None
        else:
            mask = data[mask_idx].to(device)

        if enable_multitask:
            y_disc, event_time, censor, clinical_data_list, stage_labels = data[2], data[3], data[4], data[5], data[7]
            stage_labels = stage_labels.to(device)
        else:
            y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality in ["coattn", "coattn_motcat"]:
        
        data_WSI = data[0].to(device)
        data_omic1 = data[1].type(torch.FloatTensor).to(device)
        data_omic2 = data[2].type(torch.FloatTensor).to(device)
        data_omic3 = data[3].type(torch.FloatTensor).to(device)
        data_omic4 = data[4].type(torch.FloatTensor).to(device)
        data_omic5 = data[5].type(torch.FloatTensor).to(device)
        data_omic6 = data[6].type(torch.FloatTensor).to(device)
        data_omics = [data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6]

        if enable_multitask:
            y_disc, event_time, censor, clinical_data_list, mask, stage_labels = data[7], data[8], data[9], data[10], data[11], data[12]
            stage_labels = stage_labels.to(device)
        else:
            y_disc, event_time, censor, clinical_data_list, mask = data[7], data[8], data[9], data[10], data[11]
        mask = mask.to(device)

    elif modality in ["survpath"]:

        data_WSI = data[0].to(device)

        data_omics = []
        for item in data[1][0]:
            data_omics.append(item.to(device))
        
        mask_idx = 6 if not enable_multitask else 7
        if data[mask_idx][0,0] == 1:
            mask = None
        else:
            mask = data[mask_idx].to(device)

        if enable_multitask:
            y_disc, event_time, censor, clinical_data_list, stage_labels = data[2], data[3], data[4], data[5], data[7]
            stage_labels = stage_labels.to(device)
        else:
            y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
        
    else:
        raise ValueError('Unsupported modality:', modality)
    
    y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)

    return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask, stage_labels

def _process_data_and_forward(model, modality, device, data, enable_multitask=False):
    r"""
    Depeding on the modality, process the input data and do a forward pass on the model 
    
    Args:
        - model : Pytorch model
        - modality : String
        - device : torch.device
        - data : tuple
        - enable_multitask : Boolean
    
    Returns:
        - out : torch.Tensor or tuple
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - clinical_data_list : List
        - stage_labels : torch.Tensor (if multitask enabled)
    
    """
    data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask, stage_labels = _unpack_data(
        modality, device, data, enable_multitask)

    if modality in ["coattn", "coattn_motcat"]:  
        
        out = model(
            x_path=data_WSI, 
            x_omic1=data_omics[0], 
            x_omic2=data_omics[1], 
            x_omic3=data_omics[2], 
            x_omic4=data_omics[3], 
            x_omic5=data_omics[4], 
            x_omic6=data_omics[5]
            )  

    elif modality == 'survpath':

        input_args = {"x_path": data_WSI.to(device)}
        for i in range(len(data_omics)):
            input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
        input_args["return_attn"] = False
        out = model(**input_args)
        
    else:
        out = model(
            x_text_report = clinical_data_list[-1],
            data_omics = data_omics, 
            data_WSI = data_WSI, 
            mask = mask
            )
        
    # Handle single tensor output
    # if not enable_multitask and len(out.shape) == 1:
    #     out = out.unsqueeze(0)
    
    if enable_multitask:
        return out, y_disc, event_time, censor, clinical_data_list, stage_labels
    else:
        return out, y_disc, event_time, censor, clinical_data_list


def _calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient 
    
    Args: 
        - h : torch.Tensor 
    
    Returns:
        - risk : torch.Tensor 
    
    """
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

def _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, event_time, censor, risk, clinical_data_list, 
                   all_stage_preds=None, all_stage_labels=None, stage_pred=None, stage_label=None):
    r"""
    Update the arrays with new values 
    
    Args:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - risk : torch.Tensor
        - clinical_data_list : List
        - all_stage_preds : List (optional)
        - all_stage_labels : List (optional)
        - stage_pred : torch.Tensor (optional)
        - stage_label : torch.Tensor (optional)
    
    Returns:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - all_stage_preds : List (optional)
        - all_stage_labels : List (optional)
    
    """
    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    all_clinical_data.append(clinical_data_list)
    
    if all_stage_preds is not None and stage_pred is not None:
        all_stage_preds.append(stage_pred.detach().cpu().numpy())
        all_stage_labels.append(stage_label.detach().cpu().numpy())
        return all_risk_scores, all_censorships, all_event_times, all_clinical_data, all_stage_preds, all_stage_labels
    
    return all_risk_scores, all_censorships, all_event_times, all_clinical_data

def _train_loop_survival(epoch, model, modality, loader, optimizer, scheduler, loss_fn, enable_multitask=False, use_debiase=True, use_align=False):
    r"""
    Perform one epoch of training 

    Args:
        - epoch : Int
        - model : Pytorch model
        - modality : String 
        - loader : Pytorch dataloader
        - optimizer : torch.optim
        - loss_fn : custom loss function class 
        - enable_multitask : Boolean
    
    Returns:
        - c_index : Float
        - total_loss : Float 
        - stage_accuracy : Float (if multitask enabled)
    
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss = 0.
    total_survival_loss = 0.
    total_stage_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    
    if enable_multitask:
        all_stage_preds = []
        all_stage_labels = []

    # one epoch
    for batch_idx, data in enumerate(loader):
        
        optimizer.zero_grad()

        if enable_multitask:
            output, y_disc, event_time, censor, clinical_data_list, stage_labels = _process_data_and_forward(
                model, modality, device, data, enable_multitask)
            if use_debiase:
                if use_align:
                    survival_logits, stage_logits, logits_text, logit_te, logit_tie, logits_nde, align_loss = output
                    loss, surv_loss, stage_loss = loss_fn(survival_logits, stage_logits, y_disc, event_time, censor, stage_labels, logits_text, logit_te, logits_nde)
                    # print('align_loss: ', align_loss)
                    # print('loss: ', loss)
                    loss += 0.1 * align_loss.squeeze()
                else:
                    survival_logits, stage_logits, logits_text, logit_te, logit_tie, logits_nde = output
                    loss, surv_loss, stage_loss = loss_fn(survival_logits, stage_logits, y_disc, event_time, censor, stage_labels, logits_text, logit_te, logits_nde)
                total_survival_loss += surv_loss.item()
                total_stage_loss += stage_loss.item()
                stage_pred = torch.argmax(stage_logits, dim=1)
                all_stage_preds.append(stage_pred.detach().cpu().numpy())
                all_stage_labels.append(stage_labels.detach().cpu().numpy())
                
                h = logit_tie
            else:
                survival_logits, stage_logits = output
                loss, surv_loss, stage_loss = loss_fn(survival_logits, stage_logits, y_disc, event_time, censor, stage_labels, None, None, None)
                total_survival_loss += surv_loss.item()
                total_stage_loss += stage_loss.item()
                h = survival_logits

        else:
            output, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(
                model, modality, device, data, enable_multitask)
            if use_debiase:
                if use_align:
                    survival_logits, stage_logits, logits_text, logit_te, logit_tie, logits_nde, align_loss = output
                    loss, surv_loss, stage_loss = loss_fn(survival_logits, stage_logits, y_disc, event_time, censor, stage_labels, logits_text, logit_te, logits_nde)
                    # print('align_loss: ', align_loss)
                    # print('loss: ', loss)
                    loss += 0.1 * align_loss.squeeze()
                else:
                    survival_logits, logits_text, logit_te, logit_tie, logits_nde = output
                    loss, surv_loss, stage_loss = loss_fn(survival_logits, None, y_disc, event_time, censor, None, logits_text, logit_te, logits_nde)
                total_survival_loss += surv_loss.item()
                
                h = logit_tie
            else:
                survival_logits = output
                loss, surv_loss, stage_loss = loss_fn(survival_logits, None, y_disc, event_time, censor, None, None, None, None)
                total_survival_loss += surv_loss.item()
                h = survival_logits
            # loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor) 
        
        loss_value = loss.item()
        loss = loss / y_disc.shape[0]
        
        risk, _ = _calculate_risk(h)

        all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(
            all_risk_scores, all_censorships, all_event_times, all_clinical_data, 
            event_time, censor, risk, clinical_data_list)

        total_loss += loss_value 

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (batch_idx % 100) == 0:
            print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))
    
    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    if enable_multitask:
        # total_survival_loss /= len(loader.dataset)
        # total_stage_loss /= len(loader.dataset)
        
        # # Calculate stage accuracy
        # all_stage_preds = np.concatenate(all_stage_preds, axis=0)
        # all_stage_labels = np.concatenate(all_stage_labels, axis=0)
        
        # # Only calculate accuracy for valid stage labels
        # valid_mask = all_stage_labels != -1
        # if valid_mask.sum() > 0:
        #     stage_accuracy = (all_stage_preds[valid_mask] == all_stage_labels[valid_mask]).mean()
        # else:
        stage_accuracy = 0.0
        
        print('Epoch: {}, train_loss: {:.4f}, survival_loss: {:.4f}, stage_loss: {:.4f}, train_c_index: {:.4f}, stage_acc: {:.4f}'.format(
            epoch, total_loss, total_survival_loss, total_stage_loss, c_index, stage_accuracy))
        
        return c_index, total_loss, stage_accuracy
    else:
        print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))
        return c_index, total_loss

def _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores):
    r"""
    Calculate various survival metrics 
    
    Args:
        - loader : Pytorch dataloader
        - dataset_factory : SurvivalDatasetFactory
        - survival_train : np.array
        - all_risk_scores : np.array
        - all_censorships : np.array
        - all_event_times : np.array
        - all_risk_by_bin_scores : np.array
        
    Returns:
        - c_index : Float
        - c_index_ipcw : Float
        - BS : np.array
        - IBS : Float
        - iauc : Float
    
    """
    
    data = loader.dataset.metadata["survival_months_dss"]
    bins_original = dataset_factory.bins
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])

    #---> delete the nans and corresponding elements from other arrays 
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    #<---

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc = 0., 0., 0., 0.

    # change the datatype of survival test to calculate metrics 
    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, BS, IBS, iauc
   
    # cindex2 (cindex_ipcw)
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.
    
    # brier score 
    try:
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing BS')
        BS = 0.
    
    # IBS
    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.

    # iauc
    try:
        _, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1-all_risk_by_bin_scores[:, 1:], times=which_times_to_eval_at[1:])
    except:
        print('An error occured while computing iauc')
        iauc = 0.
    
    return c_index, c_index_ipcw, BS, IBS, iauc

def _summary(dataset_factory, model, modality, loader, loss_fn, survival_train=None, enable_multitask=False, use_debiase=True, use_align=False):
    r"""
    Run a validation loop on the trained model 
    
    Args: 
        - dataset_factory : SurvivalDatasetFactory
        - model : Pytorch model
        - modality : String
        - loader : Pytorch loader
        - loss_fn : custom loss function class
        - survival_train : np.array
        - enable_multitask : Boolean
    
    Returns:
        - patient_results : dictionary
        - c_index : Float
        - c_index_ipcw : Float
        - BS : List
        - IBS : Float
        - iauc : Float
        - total_loss : Float
        - stage_accuracy : Float (if multitask enabled)

    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.
    total_survival_loss = 0.
    total_stage_loss = 0.

    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_slide_ids = []
    
    if enable_multitask:
        all_stage_preds = []
        all_stage_labels = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    with torch.no_grad():
        for data in loader:

            if enable_multitask:
                data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask, stage_labels = _unpack_data(
                    modality, device, data, enable_multitask)

                if modality in ["coattn", "coattn_motcat"]:  
                    output = model(
                        x_path=data_WSI, 
                        x_omic1=data_omics[0], 
                        x_omic2=data_omics[1], 
                        x_omic3=data_omics[2], 
                        x_omic4=data_omics[3], 
                        x_omic5=data_omics[4], 
                        x_omic6=data_omics[5]
                    )  
                elif modality == "survpath":
                    input_args = {"x_path": data_WSI.to(device)}
                    for i in range(len(data_omics)):
                        input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                    input_args["return_attn"] = False
                    output = model(**input_args)
                else:
                    output = model(
                        x_text_report = clinical_data_list[-1], # last item is text report
                        data_omics = data_omics, 
                        data_WSI = data_WSI, 
                        mask = mask
                        )
                if use_debiase:
                    if use_align:
                        survival_logits, stage_logits, logits_text, logit_te, logits_tie, logits_nde, cos_align_loss = output
                    else:
                        survival_logits, stage_logits, logits_text, logit_te, logits_tie, logits_nde = output
                    
                    # h = survival_logits
                    h = logits_tie # 改
                    loss, surv_loss, stage_loss = loss_fn(survival_logits, stage_logits, y_disc, event_time, censor, stage_labels, logits_text, logit_te,logits_nde)
                    total_survival_loss += surv_loss.item()
                    total_stage_loss += stage_loss.item()
                    
                    # Stage predictions
                    stage_pred = torch.argmax(stage_logits, dim=1)
                    all_stage_preds.append(stage_pred.detach().cpu().numpy())
                    all_stage_labels.append(stage_labels.detach().cpu().numpy())
                else:
                    survival_logits, stage_logits = output
                    h = survival_logits
                    loss, surv_loss, stage_loss = loss_fn(survival_logits, stage_logits, y_disc, event_time, censor, stage_labels, None, None, None)
                    total_survival_loss += surv_loss.item()
                    total_stage_loss += stage_loss.item()
            else:
                data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask, _ = _unpack_data(
                    modality, device, data, enable_multitask)

                if modality in ["coattn", "coattn_motcat"]:  
                    h = model(
                        x_path=data_WSI, 
                        x_omic1=data_omics[0], 
                        x_omic2=data_omics[1], 
                        x_omic3=data_omics[2], 
                        x_omic4=data_omics[3], 
                        x_omic5=data_omics[4], 
                        x_omic6=data_omics[5]
                    )  
                elif modality == "survpath":
                    input_args = {"x_path": data_WSI.to(device)}
                    for i in range(len(data_omics)):
                        input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                    input_args["return_attn"] = False
                    output = model(**input_args)
                else:
                    output = model(
                        x_text_report = clinical_data_list[-1],
                        data_omics = data_omics, 
                        data_WSI = data_WSI, 
                        mask = mask
                        )
                
                if use_debiase:
                    if use_align:
                        survival_logits, logits_text, logit_te, logits_tie, logits_nde, cos_align_loss = output
                    else:
                        survival_logits, logits_text, logit_te, logits_tie, logits_nde = output
                    
                    # h = survival_logits
                    h = logits_tie # 改
                    loss, surv_loss, stage_loss = loss_fn(survival_logits, None, y_disc, event_time, censor, None, logits_text, logit_te,logits_nde)
                    total_survival_loss += surv_loss.item()
                else:
                    survival_logits = output
                    h = survival_logits
                    loss, surv_loss, stage_loss = loss_fn(survival_logits, None, y_disc, event_time, censor, None, None, None, None)
                    total_survival_loss += surv_loss.item()
                    
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]

            risk, risk_by_bin = _calculate_risk(h)
            all_risk_by_bin_scores.append(risk_by_bin)
            
            all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(
                all_risk_scores, all_censorships, all_event_times, all_clinical_data, 
                event_time, censor, risk, clinical_data_list)
            
            all_logits.append(h.detach().cpu().numpy())
            total_loss += loss_value
            all_slide_ids.append(slide_ids.values[count])
            count += 1

    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
    patient_results = {}
    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]
        case_id = slide_id[:12]
        patient_results[case_id] = {}
        patient_results[case_id]["time"] = all_event_times[i]
        patient_results[case_id]["risk"] = all_risk_scores[i]
        patient_results[case_id]["censorship"] = all_censorships[i]
        patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]
    
    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)

    if enable_multitask:
        total_survival_loss /= len(loader.dataset)
        total_stage_loss /= len(loader.dataset)
        
        # Calculate stage accuracy
        # all_stage_preds = np.concatenate(all_stage_preds, axis=0)
        # all_stage_labels = np.concatenate(all_stage_labels, axis=0)
        
        # # Only calculate accuracy for valid stage labels
        # valid_mask = all_stage_labels != -1
        # if valid_mask.sum() > 0:
        #     stage_accuracy = (all_stage_preds[valid_mask] == all_stage_labels[valid_mask]).mean()
        # else:
        #     stage_accuracy = 0.0
        
        return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss, 0
    else:
        return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss, 0


def _get_lr_scheduler(args, optimizer, dataloader):
    scheduler_name = args.lr_scheduler
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs


    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * len(dataloader)
    else:
        warmup_steps = 0
    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    return lr_scheduler

def _step(cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader):
    r"""
    Trains the model for the set number of epochs and validates it.
    
    Args:
        - cur
        - args
        - loss_fn
        - model
        - optimizer
        - lr scheduler 
        - train_loader
        - val_loader
        
    Returns:
        - results_dict : dictionary
        - val_cindex : Float
        - val_cindex_ipcw  : Float
        - val_BS : List
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
        - val_stage_accuracy : Float (if multitask enabled)
    """

    all_survival = _extract_survival_metadata(train_loader, val_loader)
    best_c_index = 0.0
    best_model_path = os.path.join(args.results_dir, f'best_model_fold_{cur}.pth')
    
    for epoch in range(args.max_epochs):
        if args.enable_multitask:
            _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn, args.enable_multitask)
        else:
            _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn)

        if args.enable_multitask:
            results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, val_stage_accuracy = _summary(
                args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival, args.enable_multitask)
            # if val_cindex > best_c_index and best_c_index < 1.0:
            #     best_c_index = val_cindex
        else:
            results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, val_stage_accuracy = _summary(
                args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival, args.enable_multitask)
        if val_cindex > best_c_index and best_c_index < 1.0:
            best_c_index = val_cindex
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            #     'best_c_index': best_c_index,
            #     'fold': cur,
            #     'args': args
            # }, best_model_path)
            # print('model saved!')
        print('Best Val c-index: {:.4f}, Stage accuracy: {:.4f}'.format(best_c_index, val_stage_accuracy))

    return results_dict, (best_c_index, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, val_stage_accuracy)

def _train_val(datasets, cur, args):
    """   
    Performs train val test for the fold over number of epochs

    Args:
        - datasets : tuple
        - cur : Int 
        - args : argspace.Namespace 
    
    Returns:
        - results_dict : dict
        - val_cindex : Float
        - val_cindex2 : Float
        - val_BS : Float
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
        - val_stage_accuracy : Float (if multitask enabled)
    """

    #----> gets splits and summarize
    train_split, val_split = _get_splits(datasets, cur, args)
    
    #----> init loss function
    loss_fn = _init_loss_function(args)

    #----> init model
    model = _init_model(args)
    
    #---> init optimizer
    optimizer = _init_optim(args, model)

    #---> init loaders
    train_loader, val_loader = _init_loaders(args, train_split, val_split)

    # lr scheduler 
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    #---> do train val
    results_dict, metrics = _step(cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)

    return results_dict, metrics


def _step_with_train_test_results(cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader):
    """
    修改后的_step函数，同时获取训练集和测试集结果
    """
    all_survival = _extract_survival_metadata(train_loader, val_loader)
    best_c_index = 0.0
    best_test_results = None
    best_train_results = None
    best_model_path = os.path.join(args.results_dir, f'best_model_fold_{cur}.pth')
    
    for epoch in range(args.max_epochs):
        # 训练
        _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn, args.enable_multitask)

        # 在测试集上评估
        results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, val_stage_accuracy = _summary(
            args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival, args.enable_multitask)
        
        if val_cindex > best_c_index and best_c_index < 1.0:
            best_c_index = val_cindex
            best_test_results = results_dict.copy()
            
            print(f"  🔄 同时获取训练集结果...")
            train_results_dict, _, _, _, _, _, _, _ = _summary(
                args.dataset_factory, model, args.modality, train_loader, loss_fn, all_survival, args.enable_multitask)
            best_train_results = train_results_dict.copy()
            
            # 保存模型
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(), 
            #     'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            #     'best_c_index': best_c_index,
            #     'fold': cur,
            #     'args': args
            # }, best_model_path)
            
        print('Best Val c-index: {:.4f}'.format(best_c_index))
    
    return best_test_results, best_train_results, (best_c_index, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, val_stage_accuracy)

def simple_merge_and_plot_km(train_results, test_results, dataset_name, save_dir=None):
    """
    简单的合并和绘制KM图函数
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib.pyplot as plt
    
    print("🎨 绘制三种KM图进行对比...")
    
    # 1. 合并训练集和测试集数据
    all_case_ids = []
    all_times = []
    all_risks = []
    all_censorships = []
    all_sources = []
    
    # 添加训练集数据
    for case_id, data in train_results.items():
        all_case_ids.append(case_id)
        all_times.append(data['time'])
        all_risks.append(data['risk'])
        all_censorships.append(data['censorship'])
        all_sources.append('train')
    
    # 添加测试集数据
    for case_id, data in test_results.items():
        all_case_ids.append(case_id)
        all_times.append(data['time'])
        all_risks.append(data['risk'])
        all_censorships.append(data['censorship'])
        all_sources.append('test')
    
    # 创建三个KM图的对比
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    p_values = {}
    sample_counts = {}
    
    # 图1：仅测试集（推荐方法）
    p_val_test, counts_test = plot_single_km(test_results, "Test Set Only", axes[0], color_scheme='conservative')
    p_values['test_only'] = p_val_test
    sample_counts['test_only'] = counts_test
    
    # 图2：全队列（与论文对比用）
    full_results = {**train_results, **test_results}  # 简单合并
    p_val_full, counts_full = plot_single_km(full_results, "Full Cohort", axes[1], color_scheme='paper')
    p_values['full_cohort'] = p_val_full
    sample_counts['full_cohort'] = counts_full
    
    # 图3：标注训练/测试来源的详细图
    p_val_detailed = plot_detailed_km_with_sources(all_times, all_risks, all_censorships, all_sources, 
                                                  "Detailed View", axes[2])
    p_values['detailed'] = p_val_detailed
    
    # 设置总标题
    fig.suptitle(f'{dataset_name} - Kaplan-Meier Survival Analysis Comparison', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, f'{dataset_name}_km_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📁 KM对比图已保存: {save_path}")
    
    plt.show()

    return p_values, sample_counts

def plot_single_km(results_dict, title_suffix, ax, color_scheme='standard'):
    """
    绘制单个KM图
    """
    # 提取数据
    times = [data['time'] for data in results_dict.values()]
    risks = [data['risk'] for data in results_dict.values()]
    censorships = [data['censorship'] for data in results_dict.values()]
    
    # 转换为DataFrame
    import pandas as pd
    df = pd.DataFrame({
        'time': times,
        'risk': risks,
        'censorship': censorships
    })
    
    # 分组
    median_risk = np.median(df['risk'])
    df['risk_group'] = np.where(df['risk'] >= median_risk, 'High Risk', 'Low Risk')
    df['event'] = 1 - df['censorship']
    
    high_risk_data = df[df['risk_group'] == 'High Risk']
    low_risk_data = df[df['risk_group'] == 'Low Risk']
    
    # 拟合KM曲线
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    
    kmf_high.fit(high_risk_data['time'], high_risk_data['event'])
    kmf_low.fit(low_risk_data['time'], low_risk_data['event'])
    
    # 根据颜色方案设置颜色
    if color_scheme == 'conservative':
        colors = ['#2C3E50', '#34495E']  # 保守的深色
        linewidth = 2.5
    elif color_scheme == 'paper':
        colors = ['#E74C3C', '#3498DB']  # 论文风格的鲜艳色
        linewidth = 3.5
    else:
        colors = ['#E74C3C', '#3498DB']  # 标准色
        linewidth = 2.5
    
    # 绘制
    # kmf_high.plot_survival_function(ax=ax, color=colors[0], linewidth=linewidth, ci_show=True, alpha=0.8)
    # kmf_low.plot_survival_function(ax=ax, color=colors[1], linewidth=linewidth, ci_show=True, alpha=0.8)
    
    kmf_high.plot_survival_function(ax=ax, color=colors[0], linewidth=linewidth, ci_show=True, show_censors=True, alpha=0.8)
    kmf_low.plot_survival_function(ax=ax, color=colors[1], linewidth=linewidth, ci_show=True, show_censors=True, alpha=0.8)

    # 统计检验
    results = logrank_test(high_risk_data['time'], low_risk_data['time'], 
                          high_risk_data['event'], low_risk_data['event'])
    p_value = results.p_value
    
    # 设置子图
    ax.set_title(f'{title_suffix}\nLog-Rank P = {p_value:.3e}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (Years)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.legend([f'High Risk (n={len(high_risk_data)})', 
               f'Low Risk (n={len(low_risk_data)})'], 
              loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 添加方法学标识
    if color_scheme == 'conservative':
        ax.text(0.05, 0.95, '✅ Rigorous\nNo Data Leakage', 
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    elif color_scheme == 'paper':
        ax.text(0.05, 0.95, '📄 Paper Style\n⚠️ Includes Training', 
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.8))
    
    return p_value, (len(high_risk_data), len(low_risk_data))

def plot_detailed_km_with_sources(times, risks, censorships, sources, title_suffix, ax):
    """
    绘制显示训练/测试来源的详细KM图
    """
    import pandas as pd
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    
    df = pd.DataFrame({
        'time': times,
        'risk': risks,
        'censorship': censorships,
        'source': sources
    })
    
    # 分组
    median_risk = np.median(df['risk'])
    df['risk_group'] = np.where(df['risk'] >= median_risk, 'High Risk', 'Low Risk')
    df['event'] = 1 - df['censorship']
    
    # 主要的KM曲线
    high_risk_data = df[df['risk_group'] == 'High Risk']
    low_risk_data = df[df['risk_group'] == 'Low Risk']
    
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    
    kmf_high.fit(high_risk_data['time'], high_risk_data['event'])
    kmf_low.fit(low_risk_data['time'], low_risk_data['event'])
    
    # 绘制主曲线
    kmf_high.plot_survival_function(ax=ax, color='#E74C3C', linewidth=3, ci_show=False, alpha=0.9)
    kmf_low.plot_survival_function(ax=ax, color='#3498DB', linewidth=3, ci_show=False, alpha=0.9)
    
    # 添加仅测试集的参考线
    high_risk_test = high_risk_data[high_risk_data['source'] == 'test']
    low_risk_test = low_risk_data[low_risk_data['source'] == 'test']
    
    if len(high_risk_test) > 5 and len(low_risk_test) > 5:
        kmf_high_test = KaplanMeierFitter()
        kmf_low_test = KaplanMeierFitter()
        
        kmf_high_test.fit(high_risk_test['time'], high_risk_test['event'])
        kmf_low_test.fit(low_risk_test['time'], low_risk_test['event'])
        
        kmf_high_test.plot_survival_function(ax=ax, color='#E74C3C', linewidth=2, 
                                           linestyle='--', ci_show=False, alpha=0.6)
        kmf_low_test.plot_survival_function(ax=ax, color='#3498DB', linewidth=2, 
                                          linestyle='--', ci_show=False, alpha=0.6)
    
    # 统计检验
    results = logrank_test(high_risk_data['time'], low_risk_data['time'], 
                          high_risk_data['event'], low_risk_data['event'])
    p_value = results.p_value
    
    # 设置子图
    ax.set_title(f'{title_suffix}\nLog-Rank P = {p_value:.3e}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (Years)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    
    # 详细图例
    legend_labels = [
        f'High Risk All (n={len(high_risk_data)})',
        f'Low Risk All (n={len(low_risk_data)})'
    ]
    if len(high_risk_test) > 5:
        legend_labels.extend([
            f'High Risk Test (n={len(high_risk_test)})',
            f'Low Risk Test (n={len(low_risk_test)})'
        ])
    
    ax.legend(legend_labels, loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 添加数据组成信息
    train_count = sum(df['source'] == 'train')
    test_count = sum(df['source'] == 'test')
    
    ax.text(0.05, 0.95, f'Composition:\nTrain: {train_count}\nTest: {test_count}', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    return p_value

# 修改你的主训练函数，替换原来的_step调用
def _train_val_with_km_plots(datasets, cur, args):
    """
    修改后的训练验证函数，包含KM图绘制
    """
    # 获取训练和验证数据
    train_split, val_split = _get_splits(datasets, cur, args)
    
    # 初始化组件
    loss_fn = _init_loss_function(args)
    model = _init_model(args)
    optimizer = _init_optim(args, model)
    train_loader, val_loader = _init_loaders(args, train_split, val_split)
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)
    
    test_results, train_results, metrics = _step_with_train_test_results(
        cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)
    
    dataset_name = f"{args.study.upper()}-Fold{cur}"
    p_values, sample_counts = simple_merge_and_plot_km(
        train_results, test_results, dataset_name, args.results_dir)
    
    # 保存结果
    fold_results = {
        'test_results': test_results,
        'train_results': train_results,
        'metrics': metrics,
        'km_p_values': p_values,
        'sample_counts': sample_counts
    }
    
    save_path = os.path.join(args.results_dir, f'fold_{cur}_complete_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(fold_results, f)
    
    return fold_results
