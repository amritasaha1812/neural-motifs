#!/usr/bin/env bash
# Train the model without COCO pretraining
export PYTHONPATH=/dccstor/cssblr/amrita/neural-motifs
jbsub -cores 1+1 -require k80 -interactive -err err/e_train_detector.txt -out out/o_train_detector.txt -mem 150g -q x86_7d python models/train_detector.py -b 6 -lr 1e-3 -save_dir checkpoints/vgdet -nepoch 50 -ngpu 1 -nwork 1 -p 100 -clip 5
# If you want to evaluate on the frequency baseline now, run this command (replace the checkpoint with the
# best checkpoint you found).
#export CUDA_VISIBLE_DEVICES=0
#python models/eval_rel_count.py -ngpu 1 -b 6 -ckpt checkpoints/vgdet/vg-24.tar -nwork 1 -p 100 -test
#export CUDA_VISIBLE_DEVICES=1
#python models/eval_rel_count.py -ngpu 1 -b 6 -ckpt checkpoints/vgdet/vg-28.tar -nwork 1 -p 100 -test
#export CUDA_VISIBLE_DEVICES=2
#python models/eval_rel_count.py -ngpu 1 -b 6 -ckpt checkpoints/vgdet/vg-28.tar -nwork 1 -p 100 -test
#
#
