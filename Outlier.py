import numpy as np
import matplotlib.pyplot as plt 

x = [593,6546,242345,667,878,353,676,887,898,974345,545,65,63,453,5556,7676,787,689,3535,554,566,667]

outlier = []

SND = 3
mean = np.mean(x)
std = np.std(x)

for i in x:
    z_score = (i - mean)/std
    if(np.abs(z_score) > SND):
        outlier.append(i)

print(outlier)