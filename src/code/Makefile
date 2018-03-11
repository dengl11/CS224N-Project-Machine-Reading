#start: activate virture env
s:
	source activate squad

#baseline training
bt:
	python  -W ignore main.py --experiment_name=baseline --mode=train

#bidirectional training
bi:
	python  -W ignore main.py --experiment_name=bi_attn --mode=train

#bidirectional evaluation
bie:
	python  -W ignore main.py --experiment_name=bi_attn --mode=dev\
		--json_in_path ../data/dev-v1.1.json \
		--ckpt_load_dir ../experiments/bi_attn/glove100/best_checkpoint/

#co-attentional training
co:
	python  -W ignore main.py --experiment_name=co_attn --mode=train
