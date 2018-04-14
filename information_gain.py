import numpy as np
import pandas as pd
import math

data = pd.read_csv('information_gain.csv', sep = ',')
print(data)
m = len(data)

p = len(data[data.Play_tennis == 'Yes'])/m
entropy = -1*p*math.log2(p) - (1-p)*math.log2(1-p)
print("Intial Entropy = ", entropy)
min_ent = entropy

for i in data:
    if i!='Day' and i!='Play_tennis':
        set_col = set(data[i])
        ent_att = 0

        for j in set_col:
            count = len(data[(data[i] == j) & (data['Play_tennis'] == 'Yes')])
            length = len(data[data[i] == j])

            p = count/length
            if p!=0 and p!=1:
                entropy = -1*p*math.log2(p) - (1-p)*math.log2(1-p)
            else:
                entropy = 0
            ent_att = ent_att + entropy*length/m

        print("Entropy for", i, "=", ent_att)
        if ent_att < min_ent:
            min_ent = ent_att

info_gain = entropy - min_ent
print("Information gain =", info_gain)
