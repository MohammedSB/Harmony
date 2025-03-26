# [Harmony: A Joint Self-Supervised and Weakly-Supervised Framework for Learning General Purpose Visual Representations](https://arxiv.org/abs/2405.14239)

We present Harmony, a joint self-supervised and weakly-supervised method for learning generalized visual features from web-scraped data, introducing a soft loss and a text self-distillation method. Harmony outperforms previous methods and baselines across classification, segmentation, and detection tasks, highlighting how our multiple training objectives can complement each other to learn stronger visual representations. 

## Instillation
To use Harmony, you have to install the required python environment. To do that, you should first clone this GitHub repository using:

``git clone https://github.com/MohammedSB/Harmony.git``

``cd Harmony``

Then, you can install all the necessary packages. It is advised to create a virtual environment for this before.

``pip install -r requirements.txt``

## Training Harmony
To train Harmony, you can use the following base command to train on 8 V100 GPUs using a ViT base model.
```
PYTHONPATH=. python3 /ibex/user/baharoms/Harmony/Harmony/run_with_submitit.py --partition=batch --nodes=1 --ngpus=8 \
	--epochs=25 --timeout=1400 --constraint=v100 --mem=320G --arch vit_base --global_crops_scale 0.32 1.0 --local_crops_scale 0.05 0.32 --warmup_epochs=3 \
	--local_crops_number=8 --global_crops_number=2 --objective clip_ibot_mae --warmup_teacher_temp_epochs=5 \
	--act_in_head='gelu' --saveckp_freq=5  --batch_size_per_gpu 96 --pred_ratio 0 0.3 --pred_ratio_var 0 0.2  \
	--out_dim 8192 --shared_head=True --use_fp16=True --hard_labels_weight_end=0.2 --hard_labels_weight_epochs=10 --use_mlm=True --use_text_distillation=True \
	--data "CC3M:/ibex/user/baharoms/data/CC3M/cc3m" --output_dir <OUTPUT_DIR>
```
Important arguments in this command include the `objective`, `hard_labels_weight_end` and `hard_labels_weight_epochs`. The `objective` determines the main learning objectives to be used. Harmony uses CLIP, iBOT, and MAE, but you can use any combinations of these three. The other two arguments, `hard_labels_weight_end` and `hard_labels_weight_epochs` decide the end weighting of the hard label loss (where the soft CLIP weight loss is defined as 1 - `hard_labels_weight_end`), and `hard_labels_weight_epochs` determines how many epochs it takes to go from a hard label weight of 1 to `hard_labels_weight_end` using a linear scheduler. There is also an argument, `use_siglip`, that switches the from CLIP to SigLIP loss.

### Single Node Training
If you want to train Harmony on a single machine, you can use the following command.
```
PYTHONPATH=. torchrun --nproc_per_node=1 /ibex/user/baharoms/Harmony/Harmony/main_harmony.py \
	--epochs=25 --arch vit_base --global_crops_scale 0.32 1.0 --local_crops_scale 0.05 0.32 --warmup_epochs=3 \
	--local_crops_number=8 --global_crops_number=2 --objective clip_ibot_mae --warmup_teacher_temp_epochs=5 \
	--act_in_head='gelu' --saveckp_freq=5  --batch_size_per_gpu 96 --pred_ratio 0 0.3 --pred_ratio_var 0 0.2  \
	--out_dim 8192 --shared_head=True --use_fp16=True --hard_labels_weight_end=0.2 --hard_labels_weight_epochs=10 --use_mlm=True --use_text_distillation=True \
	--data "CC3M:/ibex/user/baharoms/data/CC3M/cc3m" --output_dir <OUTPUT_DIR>
```
If you are running out of GPU memory, you should decrease the `batch_size_per_gpu` and/or `local_crops_number`.

## Evaluating Harmony
You can evaluate harmony across multiple tasks, including linear probing, fine-tuning, and zero-shot classification, semantic segmentation, and object detection. For semantic segmentation and object detection, we used a different environment than the original environment used to train Harmony. You should look at iBOT and BEiT to learn about how to set those up.

### Classification
For linear probing classification, you can use the following command:
```
PYTHONPATH=. torchrun --nproc_per_node=8 Harmony/eval/eval_linear.py --epochs=100 --arch vit_base --batch_size_per_gpu 1024 --n_last_blocks 1 --avgpool_patchtokens true \
  --pretrained_weights <PATH_TO_VIT_CHECKPOINT> --data "IMAGENET:<PATH_TO_IMAGENET>" --output_dir <OUTPUT_DIR>
```
Keep in mind here that the code saves a ViT checkpoint, and a text encoder checkpoint separately. You should use the ViT checkpoint that should be called `main_vit_checkpoint.pth`.

For zero-shot classification, you can use the following command:
```
PYTHONPATH=/ibex/user/baharoms/Harmony python3 Harmony/eval/eval_zeroshot.py --output_dir=<OUTPUT_DIR> --image_encoder=<PATH_TO_VIT_CHECKPOINT> --text_encoder=<PATH_TO_TEXT_ENCODER> --arch vit_base
```
The zero-shot pipeline is similar to SLIP, so you should follow their documentation on how to set up the paths for the datasets properly.


For fine-tuning classification, you can use the following command:
```
PYTHONPATH=. python3 -m torch.distributed.launch --nnodes 1 \
    --nproc_per_node=8 \
    /ibex/user/baharoms/Harmony/Harmony/eval/classification_layer_decay/run_class_finetuning.py \
    --finetune <PATH_TO_VIT> \
    --model vit_base \
    --epochs 100 \
    --warmup_epochs 20 \
    --layer_decay 0.65 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --layer_scale_init_value 0.0 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --use_cls \
    --imagenet_default_mean_and_std \
    --output_dir <OUTPUT_DIR> \
    --data_path <PATH_TO_IMAGENET> \
    --drop_path 0.1 \
    --batch_size 128
```
The zero-shot pipeline is based on iBOT, so you can also follow their documentation for setting up the environment.

### Semantic Segmentation
For semantic segmentation, you can use the following command to train the model:
```
PYTHONPATH=. python3 -m torch.distributed.launch --nproc_per_node=4 \
  /ibex/user/baharoms/Harmony/Harmony/eval/semantic_segmentation/train.py \
  /ibex/user/baharoms/Harmony/Harmony/eval/semantic_segmentation/configs/upernet/vit_base_512_ade20k_160k.py \
  --launcher pytorch \
  --work-dir <OUTPUT_DIR> \
  --options model.backbone.use_checkpoint=True \
  model.pretrained=<PATH_TO_VIT_CHECKPOINT> \
  data.samples_per_gpu=4 \
  model.backbone.out_with_norm=true \
  optimizer.lr=8e-4
```
To evaluate that model, you can use the following command 
```
PYTHONPATH=. python3 -m torch.distributed.launch --nproc_per_node=4 \
  /ibex/user/baharoms/Harmony/Harmony/eval/semantic_segmentation/test.py \
  /ibex/user/baharoms/Harmony/Harmony/eval/semantic_segmentation/configs/upernet/vit_base_512_ade20k_160k.py \
  <PATH_TO_TRAINED_MODEL> \
  --launcher pytorch \
  --eval mIoU \
  --options model.backbone.use_checkpoint=True \
  data.samples_per_gpu=4 \
  model.backbone.out_with_norm=true \
  optimizer.lr=8e-4
```
### Object Detection
For object detection, you can use the following command:
```
PYTHONPATH=. python3 -m torch.distributed.launch --nproc_per_node 8 \
  /ibex/user/baharoms/Harmony/Harmony/eval/object_detection/train.py \
  /ibex/user/baharoms/Harmony/Harmony/eval/object_detection/configs/cascade_rcnn/vit_base_giou_4conv1f_coco_3x.py \
  --launcher pytorch \
  --work-dir <OUTPUT_DIR> \
  --deterministic \
  --cfg-options model.backbone.use_checkpoint=True \
  model.pretrained=<PATH_TO_VIT_CHECKPOINT> \
  data.samples_per_gpu=2 \
  lr_config.step=8,11 \
  runner.max_epochs=12 \
  optimizer.paramwise_cfg.layer_decay_rate=0.75
```

# Citing
If you use this repository in your work, please consider citing the following.
```
@article{harmony,
  title={Harmony: A Joint Self-Supervised and Weakly-Supervised Framework for Learning General Purpose Visual Representations},
  author={Baharoon, Mohammed and Klein, Jonathan and Michels, Dominik L},
  journal={arXiv preprint arXiv:2405.14239},
  year={2024}
}
```
