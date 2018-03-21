################ Baseline ################# 

#baseline training
bt:
	python  -W ignore main.py --attn_layer=baseline --mode=train

################# Bi-Directional ################# 

# for debug locally 
bi:
	python  -W ignore main.py\
			--attn_layer=bi_attn\
			--print_every 1\
            --batch_size 2\
			--embedding_size 50\
			--pred_layer condition\
			--mode=train\
			--dropout 0.5
bi100:
	python  -W ignore main.py \
		--mode=train\
		--context_len=300\
		--embedding_size 100\
		--attn_layer=bi_attn\
		--batch_size 60\
		--output double_lstm_dense\
		--output_size 200\
		--pred_layer condition\
		--hidden_size 300\
		--pred_hidden_sz 200\
		--learning_rate 2e-4\
		--dropout 0.28

ensumble:
	python main.py\
		--n_eval=2000\
		    --mode=eval\
			--ensumble=../experiments/ensumble 


eno:
	python main.py\
		    --mode=official_eval\
			--json_in_path=../data/dev-v1.1.json\
			--ensumble=../experiments/ensumble 

bi300:
	python  -W ignore main.py\
			--attn_layer=bi_attn\
			--context_len=300\
			--embedding_size 300\
			--mode=train\
			--dropout 0.35


#bidirectional evaluation
bie100:
	python main.py  \
		--mode=eval\
		--context_len=300\
		--embedding_size 100\
		--attn_layer=bi_attn\
		--batch_size 60\
		--output double_lstm_dense\
		--output_size 200\
		--pred_layer condition\
		--hidden_size 200\
		--pred_hidden_sz 200\
		--glove_path=../data/glove.6B.100d.txt\
		--json_in_path=../data/dev-v1.1.json\
		--ckpt_load_dir=../analysis/runs/test_emsumble/1_A_bi_attn_E_100_D_0.22_double_lstm/best_checkpoint 
							
				   
                                         
                                                                   
bie-300:
	python  -W ignore main.py --attn_layer=bi_attn\
			--embedding_size 300\
			--mode=eval\
			--json_in_path ../data/dev-v1.1.json \
			--ckpt_load_dir ../experiments/bi_attn/glove300/best_checkpoint/


################# Co-Attention ################# 


#co-attentional training
co100:
	python  -W ignore main.py \
			--embedding_size 200\
			--attn_layer=co_attn\
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
			--attn_layer=co_attn\
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
			--attn_layer=co_attn\
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
			--attn_layer=co_attn\
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
			--attn_layer=co_attn\
			--mode=eval\
			--json_in_path ../data/dev-v1.1.json \
			--ckpt_load_dir ../experiments/co_attn/glove100/best_checkpoint/




################# Self-Attention #################
#Don't change. If want to change, just create a new one.
self:
	python  -W ignore main.py \
			--embedding_size 100\
			--attn_layer=self_attn\
			--batch_size=100\
			--context_len=300\
			--hidden_size=50\
			--output_size=100\
			--output lstm\
			--dropout 0.15\
			--mode=train

self_small_learning:
	python  -W ignore main.py \
			--embedding_size 100\
			--attn_layer=self_attn\
			--batch_size=100\
			--context_len=300\
			--hidden_size=70\
			--output_size=100\
			--output lstm\
			--dropout 0.15\
			--mode=train\
			--learning_rate 3e-4


self_small_learning_double_lstm:
	python -W ignore main.py \
		--embedding_size 100\
		--attn_layer=self_attn\
		--batch_size=100\
		--context_len=300\
		--hidden_size=85\
		--output_size=100\
		--output double_lstm\
		--dropout 0.15\
		--mode=train\
		--learning_rate 3e-4


self_small_learning_double_lstm_dense:
	python -W ignore main.py \
		--embedding_size 100\
		--attn_layer=self_attn\
		--batch_size=100\
		--context_len=300\
		--hidden_size=100\
		--output_size=200\
		--output double_lstm_dense\
		--pred_layer condition\
		--pred_hidden_sz 200\
		--dropout 0.22\
		--mode=train\
		--learning_rate 7e-4\
		--decay_rate 0.75

self_bigger_learning_rate_smaller_decay_rate:
	python -W ignore main.py \
		--embedding_size 100\
		--attn_layer=self_attn\
		--batch_size=100\
		--context_len=300\
		--hidden_size=100\
		--output_size=200\
		--output double_lstm_dense\
		--pred_layer condition\
		--pred_hidden_sz 200\
		--dropout 0.25\
		--mode=train\
		--learning_rate 1e-3\
		--decay_rate 0.6

self_big:
	python  -W ignore main.py \
			--embedding_size 100\
			--attn_layer=self_attn\
			--batch_size=100\
			--context_len=500\
			--hidden_size=160\
			--output_size=200\
			--output lstm\
			--dropout 0.15\
			--mode=train


################# Combined-Attention #################
combined_small_quick_learning:
	python  -W ignore main.py \
		--embedding_size 100\
		--attn_layer=combined_attn\
		--batch_size=100\
		--context_len=300\
		--hidden_size=70\
		--output_size=200\
		--output double_lstm_dense\
		--pred_layer condition\
		--pred_hidden_sz 200\
		--dropout 0.25\
		--mode=train\
		--learning_rate 1e-3\
		--decay_rate 0.6

combined_small_learning_double_lstm_dense:
	python -W ignore main.py \
	        --embedding_size 100\
                --attn_layer=combined_attn\
                --batch_size=50\
                --context_len=300\
                --hidden_size=150\
                --output_size=200\
                --output double_lstm_dense\
                --pred_layer condition\
                --pred_hidden_sz 200\
                --dropout 0.15\
                --mode=train\
                --learning_rate 7e-4\
		--encoder lstm

combined_small_learning_with_dense_softmax_pred:
	python -W ignore main.py \
                --embedding_size 100\
                --attn_layer=combined_attn\
                --batch_size=100\
                --context_len=300\
                --hidden_size=150\
                --output_size=200\
                --output double_lstm_dense\
                --pred_layer dense+softmax\
                --pred_hidden_sz 200\
                --dropout 0.15\
                --mode=train\
                --learning_rate 7e-4\
                --encoder lstm

 combined_dense_softmax_pred_low_dropout:
	python -W ignore main.py \
                --embedding_size 100\
                --attn_layer=combined_attn\
                --batch_size=100\
                --context_len=300\
                --hidden_size=120\
                --output_size=200\
                --output double_lstm_dense\
                --pred_layer dense+softmax\
                --pred_hidden_sz 200\
                --dropout 0.09\
                --mode=train\
                --learning_rate 7e-4\
                --encoder lstm

cli:
	pip install codalabworker
	 pip install codalab

cld:
	cl work main::cs224n-DDD

clr:
	cl run --name gen-answers --request-cpus 1 --request-memory 4g --request-disk 1g --request-time 1d --request-docker-image abisee/cs224n-dfp:v4 context_idf.txt:context_idf.txt code:code best_checkpoint:best_checkpoint glove.txt:0x97c870/glove.6B.100d.txt data.json:0x8f29fe 'python code/main.py --idf_path=../context_idf.txt --mode=official_eval --context_len=300 --attn_layer bi_attn --hidden_size 300 --output double_lstm_dense --output_size 200 --pred_hidden_sz 200 --dropout 0.28 --glove_path=glove.txt --json_in_path=data.json --ckpt_load_dir=best_checkpoint'
							
