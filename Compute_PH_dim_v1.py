import numpy as np
import pandas as pd
import random
from scipy.io import savemat, loadmat
from ripser import ripser
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence

# Dataset benchmark
name_dataset = 'Ikeda'

# Setting
dim_PH = 1
dim_PH_str = '1'
alpha = 1.

print ('alpha: ', alpha)

# Dataset from Neuroscience
# reactgo fdgo contextdm1
# name_file = 'reactgo' + str('.csv')
# data_original = pd.read_csv('./Dati_neuro/' + name_file, sep='\t', header = None, decimal = '.')
# print(data_original)
# print(data_original.shape)

# data = data_original[0:N].to_numpy()

# print(data)
# print(np.shape(data))

n_k_list = []
l_k_list = []

number_list = [20000, 25000, 30000]
number_list_str = ['20000', '25000', '30000']

for i in range(3):

    number_points = number_list_str[i]
    filename_to_load = name_dataset + number_points + '_PH' + dim_PH_str
    diagrams_giotto = np.load('./PH/' + filename_to_load + '.npy', allow_pickle = True)

    print(diagrams_giotto)

    L_0 = np.sum((diagrams_giotto[0][:,1]-diagrams_giotto[0][:,0])**alpha)

    print(L_0)

    n_k_list.append(number_list[i])
    l_k_list.append(L_0)

# print(n_k_list)
# print(l_k_list)

log_n_k = np.log10(np.array(n_k_list))
log_l_k = np.log10(np.array(l_k_list))

print(log_n_k)
print(log_l_k)

plt.loglog(log_n_k, log_l_k, '-*r')
plt.show()

# print(log_n_k)
# print(log_l_k)

p = np.polyfit(log_n_k, log_l_k, 1)

polyx = np.arange(log_n_k[0], log_n_k[-1]+0.01, 0.01)
polyy = p[0]*polyx+p[1]

# alpha, C, A = extrapolate_asymptotics(np.array(n_k_list), np.array(l_k_list))

print('---------------- RESULTS -----------------')
# print('Dataset: ', name_file)
print('PH_i, i = ', dim_PH)
print('----------------------------------------')

# print('coeff_polyfit: ', p)
# print(p[0])
print('d_estimated: ', 1./(1.-p[0]))
# print(p[1])

print('----------------------------------------')
print('----------------- END --------------------')

# y_val = alpha*polyx + C

# plt.plot(polyx, y_val, 'r')
# plt.plot(log_n_k, log_l_k, 'ob')
# plt.plot(polyx, polyy)
# plt.show()

plt.plot(polyx, polyy, 'r')
plt.plot(log_n_k, log_l_k, '*b')
# plt.plot(log_n_k[0], log_l_k[0], 'og')
plt.show()