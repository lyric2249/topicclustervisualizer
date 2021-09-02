from math import *
from numpy.lib.shape_base import apply_along_axis

import pybind11

import numpy as np
import pandas as pd

from functionc import onepl_lsrm_cont_missing

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from scipy.spatial.distance import euclidean

#from src.functionp import *

#functionp.functionp

#from ../functionp/functionp import *
#import function1.lsrm1pl_normal_missing_mine

import scipy

#import sys
#sys.path.insert(0, '/directory/tothe/handshakefile/')

#sys.path

from functionp import procrustes_mine, affinityMatrix









data = pd.read_csv("C:/Users/Song1/source/repos/Project3/data_prob_90.csv")

#Mat = np.random.rand(50,50)


#data = pd.DataFrame()

data_topics = pd.Series(data.columns)
data_word = data.iloc[:, 0]
data_source = data.columns[1:]
data = data.drop(["words"], axis=1)
data = data.fillna(99)


data_m = np.array(data)
#logit_x = log(data_m / (1 - data_m))
data_m = np.log(data_m / (1 - data_m))

ndim = 2
niter = 55000
nburn = 5000
nthin = 5
nprint = 5000


"""
모든 topic의 분포는 theta ~ 디리클레 알파를 따름

biterm 총체 B에서 biterm b를 뽑으면, 이 biterm이 어느 topic z에 속할지는 z ~ 다항분포 세타

topic z의 topic-word 분포는 phi_z ~ 디리클레 베타를 따름
topic = z, word = w. z가 정해졌을 때 이로부터 각 단어가 가지는 확률을 모아서 set으로 한것이 phi_z.

골라진 topic에 대응하는 topic-word 분포로부터 단어 2개가 골라질 확률은 w_i, w_j ~ 다항(phi_z)를 따름
"""


jump_beta = 0.3
jump_theta = 1.0
jump_w = 0.06
jump_z = 0.50
jump_gamma = 0.01

pr_mean_beta = 0
pr_sd_beta = 1
pr_mean_theta = 0
pr_sd_theta = 1
pr_mean_gamma = 0.0
pr_sd_gamma = 1.0
pr_a_sigma = 0.001
pr_b_sigma = 0.001
pr_a_th_sigma = 0.001
pr_b_th_sigma = 0.001

missing = 99

# Set 99 as missing
output2 = onepl_lsrm_cont_missing(data,

                                 ndim, 
                                 niter, 
                                 nburn, 
                                 nthin, 
                                 nprint,

                                 jump_beta, 
                                 jump_theta, 
                                 jump_gamma, 
                                 jump_z, 
                                 jump_w,

                                 pr_mean_beta, 
                                 pr_sd_beta, 
                                 pr_a_th_sigma, 
                                 pr_b_th_sigma, 
                                 pr_mean_theta,
                                 pr_a_sigma, 
                                 pr_b_sigma, 
                                 pr_mean_gamma, 
                                 pr_sd_gamma,

                                 missing)

output = {}

output["beta"] = output2[0]
output["theta"] = output2[1]
output["z"] = output2[2]
output["w"] = output2[3]
output["gamma"] = output2[4]
output["sigma_theta"] = output2[5]
output["sigma"] = output2[6]
output["map"] = output2[7]
output["accept_beta"] = output2[8]
output["accept_theta"] = output2[9]
output["accept_z"] = output2[10]
output["accept_w"] = output2[11]
output["accept_gamma"] = output2[12][0]

del(output2)



################################################################################################

def MCMC_process(data):

    import time

    start = time.time()

    nsample, nitem = data.shape
    nmcmc = int((niter - nburn) / nthin)

    max_address = np.argmax(output['map'])
    
    w_star = output['w'][max_address, :, :]
    z_star = output['z'][max_address, :, :]
    
    w_proc = np.zeros((nmcmc, nitem, ndim), )
    z_proc = np.zeros((nmcmc, nsample, ndim), )

    for iter in range(nmcmc):
        z_iter = output['z'][iter, :, :]

        if iter != max_address:
            z_proc[iter, :, :] = procrustes_mine(z_iter, z_star)
        else:
            z_proc[iter, :, :] = z_iter

        w_iter = output['w'][iter, :, :]

        if iter != max_address:
            w_proc[iter, :, :] = procrustes_mine(w_iter, w_star) # TODO: ======= 210717 =======
        else:
            w_proc[iter, :, :] = w_iter

    w_est = np.empty((nitem, ndim))

    for i in range(nitem):
        for j in range(ndim):
            w_est[i, j] = w_proc[:, i, j].mean()

    z_est = np.empty((nsample, ndim,))

    for k in range(nsample):
        for j in range(ndim):
            z_est[k, j] = z_proc[:, k, j].mean()

    beta_est = output["beta"].mean()
    theta_est = output["theta"].mean()

    # beta_est = apply(output["beta"], 2, mean)
    # theta_est = apply(output["theta"], 2, mean)

    sigma_theta_est = output["sigma_theta"].mean()
    gamma_est = output["gamma"].mean()

    output_new = {"beta_estimate": beta_est,
                "theta_estimate": theta_est,
                "sigma_theta_estimate": sigma_theta_est,
                "gamma_estimate": gamma_est,
                "z_estimate": z_est,
                "w_estimate": w_est,
                "beta": output["beta"],
                "theta": output["theta"],
                "theta_sd": output["sigma_theta"],
                "gamma": output["gamma"],
                "z": z_proc,
                "w": w_proc,
                "accept_beta": output["accept_beta"],
                "accept_theta": output["accept_theta"],
                "accept_w": output["accept_w"],
                "accept_z": output["accept_z"],
                "accept_gamma": output["accept_gamma"]
                }

    print(round(time.time() - start, 2), " seconds")
    
    return output_new


################################################################################################


#procrustes_mine(z_iter, z_star)

#import inspect
#inspect.getsource(procrustes_mine)

#from functionp import procrustes_mine, affinityMatrix

output_new = MCMC_process(data)

## lsrm plot
#### ggplot

a = pd.DataFrame(output_new["z_estimate"], columns=["coordinate_1", "coordinate_2"])
b = pd.DataFrame(output_new["w_estimate"], columns=["coordinate_1", "coordinate_2"])  # x와 w의 coordinate로 df 구성

data_m

b["topic_name"] = data_topics
b["id"] = range(1, data_m.shape[1] + 1)

#### Rotate

angle = -pi / 30

M = pd.DataFrame([[cos(angle), sin(angle)], [-sin(angle), cos(angle)]], columns=["coordinate_1", "coordinate_2"])
# clockwise 회전용도. 따라서 일반적인 회전행렬의 inv. clockwise 기준 -6도씩 돌림.





bnew = pd.DataFrame(np.dot(-b.iloc[:, 0:2], M), columns = ["coordinate_1", "coordinate_2"])
#for 75, 70 # 계수 컬럼1과 컬럼2가 교환된 식이 존재했음. for what?
bnew["id"] = b["id"]
bnew["topic_name"] = b["topic_name"]
#bnew.columns = b.columns



anew = pd.DataFrame(np.dot(-a.iloc[:, 0:2], M), columns = ["coordinate_1", "coordinate_2"])

######## distance

a["dist"] = (a["coordinate_1"] ** 2 + a["coordinate_2"] ** 2) ** 0.5
a_new = a.loc[(a["dist"] > 1.5), :] #TODO: 1.5 is right?





####################### TODO:: 3dplot

new = pd.DataFrame({"x": b["coordinate_1"],
                    "y": b["coordinate_2"],
                    "z": output_new["beta_estimate"],
                    "topics": b["topic_name"]})  
# word의 계수1, word의 계수2, beta의 측정치, word의 topic name,
# output_new는 z_est, w_est, z.proc, w.proc으로 바꾼버전



word_cluster = KMeans(n_clusters = 4).fit(output_new["z_estimate"]).labels_
topic_cluster = KMeans(n_clusters = 4).fit(b.iloc[:, 1:2]).labels_

colors = topic_cluster

topic_plot = scatterplot3d(new[:, 1:3],
                           pch=16,
                           color=colors,
                           angle=50)  # TODO: =================================
# word_plot$points3d(output_new$w_estimate,pch=8,color=color2)

####################################

wcss = [1, 2]
bet_tot = [1, 2]
bet = [1, 2]
ncluster = 5

data_set2 = data_m





for aa in range(3, data_set2.shape[1]):
    # aa=3

    X = b.iloc[:, 0:2]

    idist = np.zeros((X.shape[0], X.shape[0]))

    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            idist[i, j] = euclidean(X.iloc[i, :], X.iloc[j, :])
            continue


    # idist = as.matrix(dist(x= as.data.frame(b.iloc[:, 1:2]), method = "euclidean"))  # TODO: ====================

    
    W = affinityMatrix(idist, K=aa)  # TODO: =====================
    d = W.sum(axis=1)

    d[d == 0] = np.finfo("double").eps

    D = np.diag(d)

    L = D - W

    Di = np.diag(1 / np.sqrt(d))

    NL = Di @ L @ Di
    ev = np.linalg.eig(NL)[0]
    evec = np.linalg.eig(NL)[1]

    ix = pd.Index(abs(ev)).sort_values(return_indexer=True)[1]
    U = evec[:, ix[0:ncluster]]

    U = (U.transpose()/np.sqrt(np.sum(U ** 2, 1))).transpose()



    """
    def normalize(x):
        return x / sqrt(sum(x ^ 2))

    U = pd.DataFrame(U)
    U.apply(normalize, 1)

    U = np.array([8,4,6,3]).reshape((2,2)).transpose()
    np.apply_along_axis(normalize, 1, U)
    """

    final = KMeans(n_clusters=ncluster, random_state=0).fit(U)
    group = final.labels_

    totss = sum(sum(scale(X, axis=0, with_std=False) ** 2))
    tot_withinss = final.inertia_
    betweenss = totss - tot_withinss

    wcss.append(tot_withinss)  #TODO: min #3:82%, 4: 61%, 5: 91%, 6:86%. 7:83%. 8: 85%, 9: 86%. 10: 87%
    bet_tot.append(betweenss / totss * 100)  # max
    bet.append(betweenss)

    #TODO: 터진듯. wcss 값이 괴상하게 작음

    
    # wcss[aa] < - final$tot.withinss  # min #3:82%, 4: 61%, 5: 91%, 6:86%. 7:83%. 8: 85%, 9: 86%. 10: 87%
    # bet_tot[aa] < - final$betweenss / final$totss * 100  # max
    # bet[aa] < - final$betweenss

# which.max(bet_tot)


W = affinityMatrix(idist, K = np.argmax(bet_tot)) # TODO: argmax 반환값이 어레이면 오류발생 가능성 # TODO: 

## cluster the topic using select number of cluster
# ncluster <- min(k)  # TODO: =====================
ncluster = 3
group_index = []

"""
for i in range(1, max(group) + 1):
    group_index[i] = min(np.where(group == i)) # TODO: =====================
"""

for i in range(max(group) + 1):
    ix = np.min(np.where(group == i))
    group_index.append(ix) # TODO: =====================


###############################

word_position = pd.concat([data_word, a.iloc[:, 0:2]], axis=1)

word_position["dist"] = (word_position["coordinate_1"] ** 2 + word_position["coordinate_2"] ** 2) ** 0.5
word_new = word_position.loc[word_position["dist"] > 1.4, :]


#quantile(word_position$dist, c(0.05, 0.2))
word_new = word_position[word_position["dist"] < 0.41,]

# word_new



words = data_word
close_word_index = []

## cosine similarity

temp = pd.concat([b.iloc[:, 0:2], pd.DataFrame({"group" : group})], axis=1)
# head(temp)



## center of topic group


# temp2 = data.frame(temp % > % group_by(group) % > % summarise(x=mean(coordinate_1), y=mean(coordinate_2)))


temp2 = temp.groupby("group", axis=0).agg({"coordinate_1": "mean", "coordinate_2": "mean"})
temp2.columns = ["x", "y"]

# head(temp2)

word_position = pd.concat([words, a.iloc[:, 0:2]], axis=1)
close_word_index = []
# head(word_position)

#library(ggrepel)


##### Cluster_Group near words


def euc_dist(x1, x2): sqrt(sum((x1 - x2) ** 2))


# head(word_position)


#save.image("75_result.RData")

