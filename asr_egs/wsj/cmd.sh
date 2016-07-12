export train_cmd=run.pl
export decode_cmd=run.pl
export cuda_cmd=run.pl
export mkgraph_cmd=run.pl

export cuda_cmd="slurm_comet.pl -p gpu-shared -t 48:00:00 --gpu 1"
