#!/usr/bin/env bash

# This is a script that will evaluate all models for SGCLS
export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=/dccstor/cssblr/amrita/neural-motifs
if [ $1 == "0" ]; then
    echo "EVALING THE BASELINE"
    python models/eval_rels.py -m sgcls -model motifnet -nl_obj 0 -nl_edge 0 -b 6 \
    -clip 5 -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/baseline-sgcls/vgrel-11.tar \
    -nepoch 50 -use_bias -test -cache baseline_sgcls
    python models/eval_rels.py -m predcls -model motifnet -nl_obj 0 -nl_edge 0 -b 6 \
    -clip 5 -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/baseline-sgcls/vgrel-11.tar \
    -nepoch 50 -use_bias -test -cache baseline_predcls
elif [ $1 == "1" ]; then
    echo "EVALING MESSAGE PASSING"
    python models/eval_rels.py -m sgcls -model stanford -b 6 -p 100 -lr 1e-3 -ngpu 1 -clip 5 \
    -ckpt checkpoints/stanford-sgcls/vgrel-11.tar -test -cache stanford_sgcls
    python models/eval_rels.py -m predcls -model stanford -b 6 -p 100 -lr 1e-3 -ngpu 1 -clip 5 \
    -ckpt checkpoints/stanford-sgcls/vgrel-11.tar -test -cache stanford_predcls
elif [ $1 == "2" ]; then
    echo "EVALING MOTIFNET"
    jbsub -cores 1+1 -require k80 -interactive -err err/e_eval_sgcls.txt -out out/o_eval_sgcls.txt -mem 150g -q x86_24h python models/eval_rels.py -coco_type $2 -coco_year $3 -coco_split $4 -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet-sgcls/vgrel-motifnet-sgcls.tar -nepoch 50 -use_bias -cache motifnet_sgcls_coco_$3_$4.pkl
#    jbsub -cores 1+1 -require k80 -interactive -err err/e_eval_predcls.txt -out out/o_eval_predcls.txt -mem 150g -q x86_24h python models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
#       -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -train -ckpt checkpoints/motifnet-sgcls/vgrel-motifnet-sgcls.tar -nepoch 50 -use_bias -cache motifnet_predcls
fi



