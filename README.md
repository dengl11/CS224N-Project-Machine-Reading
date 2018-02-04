# CS224N-Project-Machine-Reading

Stanford 2018 Winter
---------------

### Team 
- **Li Deng** (dengl11@stanford.edu)
- **Zhiling Huang** (zhiling@stanford.edu)



### Environment Setup

1. install Codelab-cli
```  
sudo pip install codalabworker
sudo pip install codalab
 ```  

### How to run

1. codelab setup
```  
cl work main::
cl new cs224n-<group-name>
cli work cs224n-<group-name>
cl wperm . public none
cl wperm . cs224n-win18-staff read 
cl gnew cs224n-<group-name>
cl uadd <partner> cs224n-<group-name>
cl wperm . cs224n-<group-name> all 
```  

2. train    
```  
python code/train.py 
```  

3. upload code/data to codelab
```  
cl upload <dir_name>
```  

### Reference:
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)

### Papers 
- [SQuAD Original Data](https://arxiv.org/pdf/1606.05250.pdf): dataset generation and basic model performance 
- [](https://arxiv.org/pdf/1608.07905.pdf)
