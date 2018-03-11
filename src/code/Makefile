#start: activate virture env
s:
	source activate squad

#baseline training
bt:
	python  -W ignore main.py --experiment_name=baseline --mode=train

#bidirectional training
bi:
	python  -W ignore main.py --experiment_name=bi_attn --mode=train

#co-attentional training
co:
	python  -W ignore main.py --experiment_name=co_attn --mode=train
