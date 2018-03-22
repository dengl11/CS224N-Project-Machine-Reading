import json
import numpy as np 

data = json.load(open("./eval_all.json"))
f1 = data['f1']
em = data['em']
print(np.mean(f1))
print(np.mean(em))
