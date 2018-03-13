################# Baseline ################# 

#baseline training
bt:
	python  -W ignore main.py --experiment_name=baseline --mode=train

#bidirectional training
bi:
	python  -W ignore main.py\
			--experiment_name=bi_attn\
			--embedding_size 50\
			--mode=train\
			--dropout 0.5
bi300:
	python  -W ignore main.py\
			--experiment_name=bi_attn\
			--context_len=300\
			--embedding_size 300\
			--mode=train\
			--dropout 0.35

################# Bi-Directional ################# 

#bidirectional evaluation
bie-100:
	python  -W ignore main.py --experiment_name=bi_attn\
		--embedding_size 100\
		--mode=eval\
		--json_in_path ../data/dev-v1.1.json \
		--ckpt_load_dir ../experiments/bi_attn/glove100/best_checkpoint/
bie-300:
	python  -W ignore main.py --experiment_name=bi_attn\
		--embedding_size 300\
		--mode=eval\
		--json_in_path ../data/dev-v1.1.json \
		--ckpt_load_dir ../experiments/bi_attn/glove300/best_checkpoint/


################# Co-Attention ################# 


#co-attentional training
co100:
	python  -W ignore main.py \
			--embedding_size 100\
			--experiment_name=co_attn\
			--batch_size=200\
			--context_len=360\
			--hidden_size=150\
			--output_size=100\
			--dropout 0.25\
			--mode=train

co200:
	python  -W ignore main.py \
			--embedding_size 200\
			--experiment_name=co_attn\
			--batch_size=200\
			--context_len=360\
			--hidden_size=150\
			--output_size=100\
			--dropout 0.25\
			--mode=train


#co-attentional training
coe100:
	python  -W ignore main.py \
			--embedding_size 100\
			--experiment_name=co_attn\
			--mode=eval\
			--json_in_path ../data/dev-v1.1.json \
			--ckpt_load_dir ../experiments/co_attn/glove100/best_checkpoint/ 

