
1 - 224N-Dev
104.215.90.124
———————————	
python  -W ignore main.py\
            --experiment_name=bi_attn\
            --embedding_size 100\
            --context_len=500\
            --batch_size 100\
            --output_size 200\
            --output lstm\
            --hidden_size 160\
            --pred_hidden_sz 100\
            --dropout 0.25\
            --mode=train
————————————	
Num params: 1596962



2 - cs224n-2gpu
40.124.14.101
————————————	
python  -W ignore main.py\
            --experiment_name=bi_attn\
            --embedding_size 100\
            --context_len=500\
            --batch_size 100\
            --output_size 200\
            --output lstm\
            --hidden_size 160\
            --pred_layer dense+softmax\
            --pred_hidden_sz 100\
            --mode=train\
            --dropout 0.25

————————————	
Num params: 1636462
相比于1， prediction layer为dense + softmax， 0.59->0.61



3 - cs224n-gpu-3
13.84.162.106
————————————	
python  -W ignore main.py\
            --experiment_name=bi_attn\
            --batch_size 120\
            --embedding_size 200\
            --hidden_size 160\
            --context_len=500\
            --output_size 200\
            --output lstm\
            --pred_hidden_sz 100\
            --mode=train\
            --dropout 0.25
————————————	
Num params: 1692962
相比于1，embedding 变为了200，模型dev从0.6到了0.62 但更多的overfit





4 - cs224n-gpu-2
52.151.38.118
————————————	
python  -W ignore main.py\
            --experiment_name=bi_attn\
            --embedding_size 100\
            --hidden_size 240\
            --context_len=500\
            --batch_size 100\
            --output_size 200\
            --output lstm\
            --pred_hidden_sz 100\
            --mode=train\
            --dropout 0.25
————————————	
Num params: 2349442 
相比于1， hidden size 变为了160->240， 效果不明显


