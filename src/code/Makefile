################# Baseline ################# 

#baseline training
bt:
	python  -W ignore main.py --experiment_name=baseline --mode=train

################# Bi-Directional ################# 

bi:
	python  -W ignore main.py\
			--experiment_name=bi_attn\
			--embedding_size 50\
			--mode=train\
			--dropout 0.5
bi100:
	python  -W ignore main.py\
            --experiment_name=bi_attn\
            --embedding_size 50\
            --context_len=500\
            --batch_size 100\
            --output_size 200\
            --output lstm\
            --hidden_size 160\
            --pred_hidden_sz 100\
            --mode=train\
            --dropout 0.25

bi300:
	python  -W ignore main.py\
			--experiment_name=bi_attn\
			--context_len=300\
			--embedding_size 300\
			--mode=train\
			--dropout 0.35


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
			--embedding_size 200\
			--experiment_name=co_attn\
			--batch_size=200\
			--context_len=500\
            --output_size 200\
            --output lstm\
			--hidden_size=300\
			--output_size=200\
			--dropout 0.20\
			--mode=train

co100cond:
	python  -W ignore main.py \
			--embedding_size 50\
			--experiment_name=co_attn\
			--batch_size=200\
			--context_len=500\
			--hidden_size=300\
			--pred_layer condition\
			--output_size=200\
			--dropout 0.20\
			--mode=train

co100ds:
	python  -W ignore main.py \
			--embedding_size 200\
			--experiment_name=co_attn\
			--batch_size=200\
			--context_len=480\
			--hidden_size=200\
			--pred_layer dense+softmax\
			--output_size=200\
			--dropout 0.20\
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




################# Self-Attention #################
self:
	python  -W ignore main.py \
			--embedding_size 100\
			--experiment_name=self_attn\
		    --attn_layer=self_attn\
			--batch_size=100\
			--context_len=300\
			--hidden_size=60\
			--output_size=100\
			--dropout 0.15\
			--mode=train

self_big:
	python  -W ignore main.py \
			--embedding_size 100\
			--experiment_name=self_attn\
			--attn_layer=self_attn\
			--batch_size=100\
			--context_len=500\
			--hidden_size=160\
			--output_size=200\
			--output lstm\
			--dropout 0.15\
			--mode=train

