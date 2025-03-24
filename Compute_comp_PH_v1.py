import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import gudhi
from gtda.homology import VietorisRipsPersistence
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from ripser import ripser
import random

F_i_list = []

# Sierpinski dataset
def create_Serpinski_points(n):
    
    vertex_coordinates = [[0.0, 0.0], [1.0, 0.0], [math.sqrt(3)/2., 1./2.]]
    x_coordinates = [vertex_coordinates[0][0]]
    y_coordinates = [vertex_coordinates[0][1]]
    for i in range(n):
        r = random.randrange(3)
        x_coordinates.append((x_coordinates[-1] + vertex_coordinates[r][0]) / 2)
        y_coordinates.append((y_coordinates[-1] + vertex_coordinates[r][1]) / 2)

    arr = np.column_stack((x_coordinates, y_coordinates))   

    return arr

# Benchmark dataset
index_start1 = 89
index_end1 = 114

index_list = []

for i in range(200):

    index_start = index_start1 + 115*i
    index_end = index_end1 + 115*i

    values = np.arange(index_start,index_end+1,1).tolist()

    #print('Iterazione: ', i+1, '/ Length list: ', len(index_list))

    index_list = index_list + values



################## COMPUTE PH COMPLEXITY ################

F_i_list = []

# Benchmark dataset
# data = loadmat('./Benchmark/Mist5000.mat')['data']

# Setting
# dim_PH = 1
#plt.title('Intrinsic dimension Mist2500 (ID = 4)')
# N = 23000

# Dataset from Neuroscience
# reactgo fdgo contextdm1
# name_file = 'reactgo' + str('.csv')
# data_original = pd.read_csv('./Dati_neuro/' + name_file, sep='\t', header = None, decimal = '.')
# print(data_original)
# print(data_original.shape)

# data = data_original[0:N].to_numpy()
# data = data[index_list]

# Serpinski dataset
# N = 5000
# dim_PH = 1
# data = create_Serpinski_points(N)
# diagrams = np.load('./PH/Serpinski5000_PH1.npy', allow_pickle = True)

# Other_dataset
# Dataset on the square [0,1]x[0,1]
# N = 10000
# dim_PH = 1
# data = np.random.rand(N,2)
# np.save('./Other_dataset/Serpiski5000.npy', data)
diagrams = np.load('./PH/Ikeda30000_PH0.npy', allow_pickle = True)
dim_PH = 0
#############################################################

# print('VR computation - Start')
# VR = VietorisRipsPersistence(homology_dimensions=[dim_PH])  # Parameter explained in the text
# diagrams = VR.fit_transform([data])
# print('VR computation - End')

# np.save('./PH/Serpinski5000_PH1.npy', diagrams)

abs_I = diagrams[0][:,1]-diagrams[0][:,0]

# print(abs_I)
# print(type(abs_I))
# print(np.max(abs_I))

min_val = np.min(abs_I)
max_val = np.max(abs_I)

# print(min_val)
# print(max_val)

min_val = 0.05
max_val = 0.1

eps_val = list(np.arange(min_val,max_val,(max_val-min_val)/100.))

print('min_val: ', min_val)
print('max_val: ', max_val)
print('len eps_val: ', len(eps_val))
print('eps_val: ', eps_val)

for k in range(100):

    eps = eps_val[k]

    print('Iteration: ', k)

    bool_array = abs_I > eps

    num = np.shape(bool_array[bool_array == True])[0]

    F_i_list.append(num)

print('eps_val: ', eps_val)
print('F_i_list: ', F_i_list)

# plt.figure(1)
plt.loglog(np.array(eps_val),np.asarray(F_i_list),'-*r')
plt.title('Loglog plot: eps and F_i (Ikeda30000, PH = 0)')
plt.xlabel('Epsilon')
plt.ylabel('F_i')
# plt.figure(2)
# plt.loglog(np.array(eps_val), -2 * np.array(eps_val) + 1000,'-r')
plt.show()

arr_X = np.log10(np.array(eps_val))
arr_y = np.log10(np.asarray(F_i_list))

print(np.shape(arr_X))
print(np.shape(arr_y))

p = np.polyfit(arr_X, arr_y, 1)

polyy = p[0]*arr_X+p[1]


print('---------------- RESULTS -----------------')
#print('Dataset: ', name_file)
print('PH_i, i = ', dim_PH)
print('----------------------------------------')

print('coeff_polyfit: ', p)
print('Slope: ', p[0])

print('----------------------------------------')


plt.title('comp_PH0_Ikeda30000_epsilon (' + str(min_val) + ',' + str(max_val) + ')')
plt.plot(arr_X, polyy, 'r', label = 'Interp.: Line (' + str(-p[0]) + ')') #"{:.2f}".format(-p[0]) + ')')
plt.plot(arr_X, arr_y, '*b', label = 'Points')
plt.legend(loc='upper right')
plt.xlabel('Epsilon')
plt.ylabel('F_i')
plt.savefig('comp_PH0_Ikeda30000_epsilon.png')
plt.show()