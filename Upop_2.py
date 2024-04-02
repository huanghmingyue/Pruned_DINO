# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device

import torch
from torch import nn

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def print_compression_statistics(model):
    mask_attn_list, mask_ffn_list, mask_dattn_list= [], [], []
    reserved_ratio = lambda x: (torch.count_nonzero(x) / torch.numel(x)).item()
    for i in range(model.module.transformer.encoder.num_layers):
        mask_attn_list.append(getattr(model.module.transformer.encoder.layers, str(i)).self_attn.alpha.data.view(-1))
        mask_ffn_list.append(getattr(model.module.transformer.encoder.layers, str(i)).ffn.alpha.data.view(-1))
        

    for i in range(model.module.transformer.decoder.num_layers):
        mask_attn_list.append(getattr(model.module.transformer.decoder.layers, str(i)).cross_attn.alpha.data.view(-1))
        mask_ffn_list.append(getattr(model.module.transformer.decoder.layers, str(i)).ffn.alpha.data.view(-1))
        mask_dattn_list.append(getattr(model.module.transformer.decoder.layers, str(i)).self_attn.alpha.data.view(-1))
        
    print_format = lambda x: [round(i * 100, 2) for i in x]
    print('mask_attn_list:  ', print_format([reserved_ratio(x) for x in mask_attn_list]))
    print('mask_ffn_list:  ', print_format([reserved_ratio(x) for x in mask_ffn_list]))
    print('mask_dattn_list:  ', print_format([reserved_ratio(x) for x in mask_dattn_list]))

    print('mask_attn: ', reserved_ratio(torch.cat(mask_attn_list)))
    print('mask_ffn: ', reserved_ratio(torch.cat(mask_ffn_list)))
    print('mask_dattn: ', reserved_ratio(torch.cat(mask_dattn_list)))

def get_sparsity_loss(model):
    sparsity_loss_attn, sparsity_loss_ffn, sparsity_loss_d_attn= 0, 0, 0
    for i in range(model.module.transformer.encoder.num_layers):
        sparsity_loss_attn += torch.sum(torch.abs(getattr(model.module.transformer.encoder.layers, str(i)).self_attn.alpha))
        sparsity_loss_ffn += torch.sum(torch.abs(getattr(model.module.transformer.encoder.layers, str(i)).ffn.alpha))
    for i in range(model.module.transformer.decoder.num_layers):
        sparsity_loss_attn += torch.sum(torch.abs(getattr(model.module.transformer.decoder.layers, str(i)).cross_attn.alpha))
        sparsity_loss_ffn += torch.sum(torch.abs(getattr(model.module.transformer.decoder.layers, str(i)).ffn.alpha))
        sparsity_loss_d_attn += torch.sum(torch.abs(getattr(model.module.transformer.decoder.layers, str(i)).self_attn.alpha))
    return sparsity_loss_attn, sparsity_loss_ffn, sparsity_loss_d_attn



def compress(model, search_model):

    for i in range(search_model.transformer.encoder.num_layers):

        # encoder ffn
        in_features = getattr(model.transformer.encoder.layers, str(i)).ffn.linear1.weight.shape[-1]
        out_features = getattr(model.transformer.encoder.layers, str(i)).ffn.linear2.weight.shape[0]

        alpha = torch.squeeze(getattr(search_model.transformer.encoder.layers, str(i)).ffn.alpha.data)
        hidden_features = torch.count_nonzero(alpha)
        getattr(model.transformer.encoder.layers, str(i)).ffn.linear1 = nn.Linear(in_features, hidden_features)
        getattr(model.transformer.encoder.layers, str(i)).ffn.linear1.weight.data = getattr(search_model.transformer.encoder.layers, str(i)).ffn.linear1.weight.data[alpha==1,:]
        getattr(model.transformer.encoder.layers, str(i)).ffn.linear1.bias.data = getattr(search_model.transformer.encoder.layers, str(i)).ffn.linear1.bias.data[alpha==1]
        getattr(model.transformer.encoder.layers, str(i)).ffn.linear2 = nn.Linear(hidden_features, out_features)
        getattr(model.transformer.encoder.layers, str(i)).ffn.linear2.weight.data = getattr(search_model.transformer.encoder.layers, str(i)).ffn.linear2.weight.data[:, alpha==1]
        getattr(model.transformer.encoder.layers, str(i)).ffn.linear2.bias.data = getattr(search_model.transformer.encoder.layers, str(i)).ffn.linear2.bias.data

        # encoder attn
        in_features = getattr(model.transformer.encoder.layers, str(i)).self_attn.value_proj.weight.shape[-1]
        out_features = getattr(model.transformer.encoder.layers, str(i)).self_attn.output_proj.weight.shape[0]

        alpha = torch.squeeze(getattr(search_model.transformer.encoder.layers, str(i)).self_attn.alpha.data)
        hidden_features = torch.count_nonzero(alpha)
        parameter_ratio = 8
        getattr(model.transformer.encoder.layers, str(i)).self_attn.value_proj = nn.Linear(in_features, hidden_features*parameter_ratio)
        getattr(model.transformer.encoder.layers, str(i)).self_attn.value_proj.weight.data = \
            getattr(search_model.transformer.encoder.layers, str(i)).self_attn.value_proj.weight.data[alpha.repeat(parameter_ratio)==1,:]
        getattr(model.transformer.encoder.layers, str(i)).self_attn.value_proj.bias.data = getattr(search_model.transformer.encoder.layers, str(i)).self_attn.value_proj.bias.data[alpha.repeat(parameter_ratio)==1]
       
        getattr(model.transformer.encoder.layers, str(i)).self_attn.output_proj = nn.Linear(hidden_features, out_features)
        getattr(model.transformer.encoder.layers, str(i)).self_attn.output_proj.weight.data = \
            getattr(search_model.transformer.encoder.layers, str(i)).self_attn.output_proj.weight.data[:, alpha.repeat(parameter_ratio)==1]
        getattr(model.transformer.encoder.layers, str(i)).self_attn.output_proj.bias.data = getattr(search_model.transformer.encoder.layers, str(i)).self_attn.output_proj.bias.data

    for i in range(search_model.transformer.decoder.num_layers):

        # decoder ffn
        in_features = getattr(model.transformer.decoder.layers, str(i)).ffn.linear1.weight.shape[-1]
        out_features = getattr(model.transformer.decoder.layers, str(i)).ffn.linear2.weight.shape[0]
        alpha = torch.squeeze(getattr(search_model.transformer.decoder.layers, str(i)).ffn.alpha.data)
        hidden_features = torch.count_nonzero(alpha)
        getattr(model.transformer.decoder.layers, str(i)).ffn.linear1 = nn.Linear(in_features, hidden_features)
        getattr(model.transformer.decoder.layers, str(i)).ffn.linear1.weight.data = getattr(search_model.transformer.decoder.layers, str(i)).ffn.linear1.weight.data[alpha==1,:]
        getattr(model.transformer.decoder.layers, str(i)).ffn.linear1.bias.data = getattr(search_model.transformer.decoder.layers, str(i)).ffn.linear1.bias.data[alpha==1]
        getattr(model.transformer.decoder.layers, str(i)).ffn.linear2 = nn.Linear(hidden_features, out_features)
        getattr(model.transformer.decoder.layers, str(i)).ffn.linear2.weight.data = getattr(search_model.transformer.decoder.layers, str(i)).ffn.linear2.weight.data[:, alpha==1]
        getattr(model.transformer.decoder.layers, str(i)).ffn.linear2.bias.data = getattr(search_model.transformer.decoder.layers, str(i)).ffn.linear2.bias.data
        
        # decoder cross attn
        in_features = getattr(model.transformer.decoder.layers, str(i)).cross_attn.value_proj.weight.shape[-1]
        out_features = getattr(model.transformer.decoder.layers, str(i)).cross_attn.output_proj.weight.shape[0]
        alpha = torch.squeeze(getattr(search_model.transformer.decoder.layers, str(i)).cross_attn.alpha.data)
        hidden_features = torch.count_nonzero(alpha)
        parameter_ratio = 8
        getattr(model.transformer.decoder.layers, str(i)).cross_attn.value_proj = nn.Linear(in_features, hidden_features*parameter_ratio)
        getattr(model.transformer.decoder.layers, str(i)).cross_attn.value_proj.weight.data = \
            getattr(search_model.transformer.decoder.layers, str(i)).cross_attn.value_proj.weight.data[alpha.repeat(parameter_ratio)==1,:]
        getattr(model.transformer.decoder.layers, str(i)).cross_attn.value_proj.bias.data = getattr(search_model.transformer.decoder.layers, str(i)).cross_attn.value_proj.bias.data[alpha.repeat(parameter_ratio)==1]
        
        getattr(model.transformer.decoder.layers, str(i)).cross_attn.output_proj = nn.Linear(hidden_features, out_features)
        getattr(model.transformer.decoder.layers, str(i)).cross_attn.output_proj.weight.data = \
            getattr(search_model.transformer.decoder.layers, str(i)).cross_attn.output_proj.weight.data[:, alpha.repeat(parameter_ratio)==1]
        getattr(model.transformer.decoder.layers, str(i)).cross_attn.output_proj.bias.data = getattr(search_model.transformer.decoder.layers, str(i)).cross_attn.output_proj.bias.data

        # decoder self attn
        in_features = getattr(model.transformer.decoder.layers, str(i)).self_attn.in_proj_weight.shape[-1]
        out_features = getattr(model.transformer.decoder.layers, str(i)).self_attn.out_proj.weight.shape[0]
        alpha = torch.squeeze(getattr(search_model.transformer.decoder.layers, str(i)).self_attn.alpha.data)
        hidden_features = torch.count_nonzero(alpha)
        parameter_ratio = 3*8
       
        getattr(model.transformer.decoder.layers, str(i)).self_attn.in_proj_weight = nn.Parameter(torch.empty((hidden_features*parameter_ratio, in_features)))
        getattr(model.transformer.decoder.layers, str(i)).self_attn.in_proj_weight.data = \
            getattr(search_model.transformer.decoder.layers, str(i)).self_attn.in_proj_weight.data[alpha.repeat(parameter_ratio)==1,:]
        getattr(model.transformer.decoder.layers, str(i)).self_attn.in_proj_bias.data = \
            getattr(search_model.transformer.decoder.layers, str(i)).self_attn.in_proj_bias.data[alpha.repeat(parameter_ratio)==1]
        getattr(model.transformer.decoder.layers, str(i)).self_attn.out_proj = nn.Linear(hidden_features, out_features)
        getattr(model.transformer.decoder.layers, str(i)).self_attn.out_proj.weight.data = \
            getattr(search_model.transformer.decoder.layers, str(i)).self_attn.out_proj.weight.data[:, alpha.repeat(parameter_ratio//3)==1]
        getattr(model.transformer.decoder.layers, str(i)).self_attn.out_proj.bias.data = getattr(search_model.transformer.decoder.layers, str(i)).self_attn.out_proj.bias.data


def update_alpha_parameters(model, p, pi, print_info):
    encoder_layers = model.module.transformer.encoder.num_layers
    encoder_alpha_grad_attn = torch.stack([getattr(model.module.transformer.encoder.layers, str(i)).self_attn.alpha.grad for i in range(encoder_layers)])
    encoder_alpha_grad_ffn = torch.stack([getattr(model.module.transformer.encoder.layers, str(i)).ffn.alpha.grad for i in range(encoder_layers)])
    
    decoder_layers = model.module.transformer.decoder.num_layers
    decoder_alpha_grad_attn = torch.stack([getattr(model.module.transformer.decoder.layers, str(i)).cross_attn.alpha.grad for i in range(decoder_layers)])
    decoder_alpha_grad_ffn = torch.stack([getattr(model.module.transformer.decoder.layers, str(i)).ffn.alpha.grad for i in range(decoder_layers)])

    decoder_alpha_grad_self_attn = torch.stack([getattr(model.module.transformer.decoder.layers, str(i)).self_attn.alpha.grad for i in range(decoder_layers)])

    alpha_grad_attn = torch.cat([encoder_alpha_grad_attn, decoder_alpha_grad_attn], dim=0)
    alpha_grad_ffn = torch.cat([encoder_alpha_grad_ffn, decoder_alpha_grad_ffn], dim=0)
    
    alpha_grad_all_attn = torch.cat([alpha_grad_attn.view(-1), decoder_alpha_grad_self_attn.view(-1)])


    standarlization = lambda x: (x - torch.mean(x)) / torch.std(x)
    alpha_grad_attn, alpha_grad_ffn = standarlization(alpha_grad_attn), standarlization(alpha_grad_ffn)
    decoder_alpha_grad_self_attn = standarlization(decoder_alpha_grad_self_attn)
    
    #把alpha_grad中的decoder_alpha_grad_self_attn与alpha_grad_ffn调换位置
    alpha_grad = torch.cat([alpha_grad_attn.view(-1), decoder_alpha_grad_self_attn.view(-1), alpha_grad_ffn.view(-1)])
    sorted_alpha_grad, indices = torch.sort(alpha_grad, descending=True)         
    compression_weight = torch.ones_like(indices)
    
    #deformable attn和multiattn的weight分开设置
    compression_weight[indices < alpha_grad_attn.numel()] = 8
    compression_weight[(indices >= alpha_grad_attn.numel()) & (indices < alpha_grad_all_attn.numel())] = 24
    
    threshold = sorted_alpha_grad[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*pi))]
    
    def update(module, grad):
        mask = ((grad <= threshold) | (grad <= torch.min(grad)))
        module.data.copy_(mask + (~mask)*(1 - pi/p))

    for i in range(encoder_layers):
        update(getattr(model.module.transformer.encoder.layers, str(i)).self_attn.alpha, alpha_grad_attn[i])
        update(getattr(model.module.transformer.encoder.layers, str(i)).ffn.alpha, alpha_grad_ffn[i])
        
    for i in range(decoder_layers):
        update(getattr(model.module.transformer.decoder.layers, str(i)).cross_attn.alpha, alpha_grad_attn[encoder_layers+i])
        update(getattr(model.module.transformer.decoder.layers, str(i)).ffn.alpha, alpha_grad_ffn[encoder_layers+i])
        update(getattr(model.module.transformer.decoder.layers, str(i)).self_attn.alpha, decoder_alpha_grad_self_attn[i])

    if print_info:
        attn, ffn, d_attn = [], [], []
        for i in range(encoder_layers):
            attn.append(getattr(model.module.transformer.encoder.layers, str(i)).self_attn.alpha.flatten())
            ffn.append(getattr(model.module.transformer.encoder.layers, str(i)).ffn.alpha.flatten())
            
        for i in range(decoder_layers):
            attn.append(getattr(model.module.transformer.decoder.layers, str(i)).cross_attn.alpha.flatten())
            ffn.append(getattr(model.module.transformer.decoder.layers, str(i)).ffn.alpha.flatten())
            d_attn.append(getattr(model.module.transformer.decoder.layers, str(i)).self_attn.alpha.flatten())
    
        print('Current compression ratio of attn: ', 1-torch.mean(torch.cat(attn)))
        print('Current compression ratio of ffn: ', 1-torch.mean(torch.cat(ffn)))
        print('Current compression ratio of d_attn: ', 1-torch.mean(torch.cat(d_attn)))
        print('Current compression ratio: ', pi)

# 1.在def train one epoch()括号里添加“search=False, update_alpha=True”
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None,
                    search=False, update_alpha=True):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train() # 将模型设置为训练模式
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    
    # 2.------------------添加以下 if语句----------------------------------------------------------------------
    ###########################################添加loss_sp_ffn################################################
    if search:
        metric_logger.add_meter('loss_sp_attn', utils.SmoothedValue(window_size=10, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_sp_ffn', utils.SmoothedValue(window_size=10, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_sp_dattn', utils.SmoothedValue(window_size=10, fmt='{value:.4f}'))
        # dattn删去
    # -------------------------------------------------------------------------------------------------
    header = 'Train Epoch: [{}]'.format(epoch) if not search else 'Search Epoch: [{}]'.format(epoch)
   # 3. 更改header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = 10
    # 4. ------------------添加以下--------------------------------------------------------------------
    len_data_loader = len(data_loader)
    total_steps = len_data_loader*args.epochs if not search else len_data_loader*args.epochs_search
    # --------------------------------------------------------------------------------------------------------

    _cnt = 0
    # 5. 更改 for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
    
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header, logger=logger)):   
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
        
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        # ---------------------------------------------------------------------------------------------------
        if search:
            sparsity_loss_attn, sparsity_loss_ffn, sparsity_loss_d_attn = get_sparsity_loss(model)
            metric_logger.update(loss_sp_ffn=args.w_sp_ffn * sparsity_loss_ffn.item()) 
            metric_logger.update(loss_sp_attn=args.w_sp_attn * sparsity_loss_attn.item()) 
            metric_logger.update(loss_sp_dattn=args.w_sp_dattn * sparsity_loss_d_attn.item())
            losses_reduced_scaled += args.w_sp_attn * sparsity_loss_attn + args.w_sp_ffn * sparsity_loss_ffn + args.w_sp_dattn * sparsity_loss_d_attn
            step = epoch*len_data_loader+i
        # ---------------------------------------------------------------------------------------

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            
# ----------------------------------------------------------------------------------------------------------------------------
            if search and ((step > 0) and (step % args.interval == 0 or step == total_steps - 1)) and update_alpha:
                pi = args.p*((1-math.cos(math.pi*(step+1)/total_steps))/2)**(1/2)
                update_alpha_parameters(model, args.p, pi, True)
# ----------------------------------------------------------------------------------------------------------------------------
        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat
    
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            # outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']


            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]



    return stats, coco_evaluator

import io, os
def prune_if_compressed(model, client, url_or_filename):
        
    if client is not None:
        with io.BytesIO(client.get(os.path.join('s3://BucketName/ProjectName', url_or_filename), enable_cache=True)) as f:
            checkpoint = torch.load(f, map_location='cpu')
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    state_dict = checkpoint['model']

    # encoder
    for i in range(model.transformer.encoder.num_layers):
        # encoder ffn
        if getattr(model.transformer.encoder.layers, str(i)).ffn.linear1.weight.shape != state_dict['transformer.encoder.layers.'+str(i)+'.ffn.linear1.weight'].shape:
            del getattr(model.transformer.encoder.layers, str(i)).ffn.linear1
            getattr(model.transformer.encoder.layers, str(i)).ffn.linear1 = nn.Linear(*state_dict['transformer.encoder.layers.'+str(i)+'.ffn.linear1.weight'].shape[::-1])
            del getattr(model.transformer.encoder.layers, str(i)).ffn.linear2
            getattr(model.transformer.encoder.layers, str(i)).ffn.linear2 = nn.Linear(*state_dict['transformer.encoder.layers.'+str(i)+'.ffn.linear2.weight'].shape[::-1])

        # encoder attn
        if getattr(model.transformer.encoder.layers, str(i)).self_attn.value_proj.weight.shape != state_dict['transformer.encoder.layers.'+str(i)+'.self_attn.value_proj.weight'].shape:
            del getattr(model.transformer.encoder.layers, str(i)).self_attn.value_proj
            getattr(model.transformer.encoder.layers, str(i)).self_attn.value_proj = nn.Linear(*state_dict['transformer.encoder.layers.'+str(i)+'.self_attn.value_proj.weight'].shape[::-1])
            del getattr(model.transformer.encoder.layers, str(i)).self_attn.output_proj
            getattr(model.transformer.encoder.layers, str(i)).self_attn.output_proj = nn.Linear(*state_dict['transformer.encoder.layers.'+str(i)+'.self_attn.output_proj.weight'].shape[::-1])

    # decoder
    for i in range(model.transformer.decoder.num_layers):
        # decoder ffn
        if getattr(model.transformer.decoder.layers, str(i)).ffn.linear1.weight.shape != state_dict['transformer.decoder.layers.'+str(i)+'.ffn.linear1.weight'].shape:
            del getattr(model.transformer.decoder.layers, str(i)).ffn.linear1
            getattr(model.transformer.decoder.layers, str(i)).ffn.linear1 = nn.Linear(*state_dict['transformer.decoder.layers.'+str(i)+'.ffn.linear1.weight'].shape[::-1])
            del getattr(model.transformer.decoder.layers, str(i)).ffn.linear2
            getattr(model.transformer.decoder.layers, str(i)).ffn.linear2 = nn.Linear(*state_dict['transformer.decoder.layers.'+str(i)+'.ffn.linear2.weight'].shape[::-1])

        # decoder cross attn
        if getattr(model.transformer.decoder.layers, str(i)).cross_attn.value_proj.weight.shape != state_dict['transformer.decoder.layers.'+str(i)+'.cross_attn.value_proj.weight'].shape:
            del getattr(model.transformer.decoder.layers, str(i)).cross_attn.value_proj
            getattr(model.transformer.decoder.layers, str(i)).cross_attn.value_proj = nn.Linear(*state_dict['transformer.decoder.layers.'+str(i)+'.cross_attn.value_proj.weight'].shape[::-1])
            del getattr(model.transformer.decoder.layers, str(i)).cross_attn.output_proj
            getattr(model.transformer.decoder.layers, str(i)).cross_attn.output_proj = nn.Linear(*state_dict['transformer.decoder.layers.'+str(i)+'.cross_attn.output_proj.weight'].shape[::-1])

        # decoder self attn

        if getattr(model.transformer.decoder.layers, str(i)).self_attn.in_proj_weight.shape != state_dict['transformer.decoder.layers.' + str(i) + '.self_attn.in_proj_weight'].shape:
            # vit attn 
            getattr(model.transformer.decoder.layers, str(i)).self_attn.in_proj_weight = nn.Parameter(torch.empty(*state_dict['transformer.decoder.layers.' + str(i) + '.self_attn.in_proj_weight'].shape))
            getattr(model.transformer.decoder.layers, str(i)).self_attn.in_proj_bias = nn.Parameter(torch.empty(*state_dict['transformer.decoder.layers.' + str(i) + '.self_attn.in_proj_bias'].shape))
            getattr(model.transformer.decoder.layers, str(i)).self_attn.out_proj = nn.Linear(*state_dict['transformer.decoder.layers.' + str(i) + '.self_attn.out_proj.weight'].shape[::-1])
        




    torch.cuda.empty_cache()
    model.load_state_dict(state_dict, strict=False)
