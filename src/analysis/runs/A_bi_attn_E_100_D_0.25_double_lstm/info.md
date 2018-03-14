
	python  -W ignore main.py\
			--attn_layer=bi_attn\
            --embedding_size 100\
            --context_len=400\
            --batch_size 70\
            --output_size 200\
			--learning_rate 8e-4\
            --output double_lstm\
            --hidden_size 200\
            --pred_hidden_sz 100\
            --mode=train\
            --dropout 0.25



[main] Epoch 8, Iter 9500, Train F1 score: 0.757295,            Train EM score: 0.618000
[main] Epoch 8, Iter 9500, Dev F1 score: 0.659539,          Dev EM score: 0.509287
