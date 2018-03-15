
	python  -W ignore main.py\
			--attn_layer=bi_attn\
            --embedding_size 300\
            --context_len=400\
            --batch_size 70\
            --output_size 200\
			--learning_rate 5e-4\
            --output lstm\
            --hidden_size 200\
			--pred_layer condition\
            --pred_hidden_sz 200\
            --mode=train\
            --dropout 0.25

[main] Epoch 14, Iter 16500, Train F1 score: 0.842830,                            Train EM score: 0.704000
[main] Epoch 14, Iter 16500, Dev F1 score: 0.658971,                            Dev EM score: 0.511115
