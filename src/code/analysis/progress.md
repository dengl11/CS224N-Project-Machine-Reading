                F1    EM
Bi-50   Train  
        Dev    0.47  0.35

Bi-300  Train  0.75  0.75  
        Dev    0.46  0.46

---------------------------------
```
bi300:
			--embedding_size 300\
			--experiment_name=bi_attn\
			--context_len=400\
			--dropout 0.24
```
#### Train 0.7 Dev 0.46
---------------------------------

---------------------------------
```
bi300:
			--experiment_name=bi_attn\
			--context_len=300\
			--embedding_size 300\
			--mode=train\
			--dropout 0.35
```
#### Train 0.58 Dev 0.42
---------------------------------

---------------------------------
```
bi300:
			--experiment_name=bi_attn\
			--embedding_size 300\
			--context_len=400\
			--dropout 0.35
```
#### Train 0.55 Dev 0.42 (11k)
---------------------------------

---------------------------------
```
co:
			--embedding_size 100\
			--experiment_name=co_attn\
			--mode=train
```
#### Train 0.71 Dev 0.51
---------------------------------

---------------------------------
```
co:
			--embedding_size 100\
			--experiment_name=co_attn\
			--context_len=400\
			--batch_size=180\
			--context_len=300\
			--dropout 0.25\
			--mode=train
```
#### Train 0.52 Dev 0.38 (11k)
---------------------------------


---------------------------------
```
co100:
			--embedding_size 100\
			--experiment_name=co_attn\
			--batch_size=200\
			--context_len=360\
			--hidden_size=150\
			--output_size=100\
			--dropout 0.25\
```
#### Train 0.57 Dev 0.39 
---------------------------------
