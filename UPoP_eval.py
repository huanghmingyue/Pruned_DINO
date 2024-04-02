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

        with torch.cuda.amp.autocast(enabled=True):
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
        
        
        _cnt += 1
        
    
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