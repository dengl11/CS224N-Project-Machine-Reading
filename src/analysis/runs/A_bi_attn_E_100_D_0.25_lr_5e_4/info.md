bi100:
	python  -W ignore main.py\
			--attn_layer=bi_attn\
            --embedding_size 100\
            --context_len=400\
            --batch_size 70\
            --output_size 200\
			--learning_rate 5e-4\
            --output lstm\
            --hidden_size 200\
			--pred_layer condition\
            --pred_hidden_sz 100\
            --mode=train\
            --dropout 0.25


[main] Epoch 14, Iter 16000, Train F1 score: 0.755300,                Train EM score: 0.617000
[main] Epoch 14, Iter 16000, Dev F1 score: 0.652629,              Dev EM score: 0.502358
