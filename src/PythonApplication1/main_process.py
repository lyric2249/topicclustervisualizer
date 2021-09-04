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


################################################################################################


#procrustes_mine(z_iter, z_star)

#import inspect
#inspect.getsource(procrustes_mine)

#from functionp import procrustes_mine, affinityMatrix

output_new = MCMC_process(data)

################################################################################################

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

hori = 5

################################################################################################

for subject in ['beta', 'theta']:
    
    fig = make_subplots(rows = 1 + int(output_new[subject].shape[1]/hori), cols = hori)
    
    for i in range(output_new[subject].shape[1]):
        
        fig.append_trace(
            go.Scatter(x = list(range(output_new[subject].shape[0])), 
                       y = output_new[subject][:,i], 
                       mode='lines'
                       ),
            row = 1 + int(i/hori),
            col = 1 + i - hori * int(i/hori)
        )
        
    #plt.subplots()
    
    fig.update_layout(
    height = 190 * (1 + int(output_new[subject].shape[1]/hori)),
    title_text="trace_" + subject
    )
    
    fig.show() 


for subject in ['theta_sd', 'gamma']:
    
    fig = px.line(x = list(range(output_new[subject].shape[0])), 
                  y = np.concatenate(output_new[subject]), 
                  title="trace_" + subject)

    fig.show() 




for subject in [['theta', 'z'], ['beta', 'w']]:
    
    for k in [0, 1]:
        
        fig = make_subplots(rows = 1 + int(output_new[subject[0]].shape[1]/hori), cols = hori)
        
        for j in range(output_new[subject[0]].shape[1]):
            #plt.title("trace_z")
            #plt.title("trace_w")

            fig.append_trace(
                go.Scatter(x = list(range(output_new[subject[1]].shape[0])), 
                           y = output_new[subject[1]][:, j, k],
                           mode='lines'
                           ),
                row = 1 + int(j/hori),
                col = 1 + j - hori * int(j/hori)
                )
            
        fig.update_layout(
            height = 190 * (1 + int(output_new[subject[0]].shape[1]/hori)),
            title_text = "trace_" + subject[1] + "_" + str(k)
            )

        fig.show()






################################################################################################




## lsrm plot
#### ggplot

a = pd.DataFrame(output_new["z_estimate"], columns=["coordinate_1", "coordinate_2"])
b = pd.DataFrame(output_new["w_estimate"], columns=["coordinate_1", "coordinate_2"])  # x와 w의 coordinate로 df 구성

data_m

b["topic_name"] = data_topics
b["id"] = range(1, data_m.shape[1] + 1)





################################################################################################



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
#for 75, 70 # 계수 컬럼1과 컬럼2가 교환된 식이 존재했음. for what?




import matplotlib.pyplot as plt
from plotnine import ggplot, geom_text

bnew
anew

for a in "a":
    fig = px.scatter(bnew, x = "coordinate_1", y = "coordinate_2", 
                    text="id")
    fig.update_traces(textposition='top center')
    fig.show()
    
    fig = px.scatter(b, x = "coordinate_1", y = "coordinate_2", 
                    text="topic_name")
    fig.update_traces(textposition='top center')
    fig.show()
    
    fig = px.scatter(anew, x = "coordinate_1", y = "coordinate_2")
    fig.update_traces(textposition='top center')
    fig.show()
    
    


"""
for i in 1:
    plt.scatter(data = bnew, )
    plt.legend("id")
    plt.scatter(data = b, x = "coordinate_1", y = "coordinate_2")
    plt.legend("topic_name")
    plt.scatter(data = anew, x = "coordinate_1", y = "coordinate_2")
    
    ### for i, txt in enumerate(df["id"]):
        ### plt.annotate(txt, (df["coordinate_1"].iat[i], df["coordinate_2"].iat[i]))
        ### plt.annotate(txt, (df.x.iat[i], df.y.iat[i]))
    ### plt.xlim(np.min(bnew["coordinate_1"].min(), anew["coordinate_1"].min()) - 0.2,
    ###          np.max(bnew["coordinate_1"].max(), anew["coordinate_1"].max()) + 0.2)
    ### plt.ylim(np.min(bnew["coordinate_2"].min(), anew["coordinate_2"].min()) - 0.2, 
    ###          np.max(bnew["coordinate_2"].max(), anew["coordinate_2"].max()) + 0.2)
    plt.show()

bnew.head()
b.head()
anew.head()
"""

"""
ggplot() + 
  geom_text(data = bnew, aes(x="coordinate_1", y = "coordinate_2", label = "id")) +
  geom_text(data = b, aes(x="coordinate_1", y = "coordinate_2", label = "topic_name")) +  #topic name
  geom_point(data = anew, aes(x="coordinate_1", y = "coordinate_2"), cex=1) + 
  xlim (min(bnew["coordinate_1"], anew["coordinate_1"])-0.2, 
        max(bnew$coordinate_1,anew$coordinate_1)+0.2) + 
  ylim(min(bnew["coordinate_2"], anew["coordinate_2"])-0.2, 
       max(bnew$coordinate_2,anew$coordinate_2)+0.2)
"""



b = bnew
a = anew




for a in "a":
    fig = px.scatter(b, x = "coordinate_1", y = "coordinate_2", 
                    text="id", title = "plot_wz")
    fig.update_traces(textposition='top center')
    fig.show()
    
    fig = px.scatter(b, x = "coordinate_1", y = "coordinate_2", 
                    text="id", title = "topic_name")
    fig.update_traces(textposition='top center')
    fig.show()
    
    
    


plt.title()
plt.scatter()

"""
pdf(paste0(dir,"/plot_wz.pdf"))

ggplot() + 
  geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = id),col=2) +
  geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = topic_name),col=2) +  #topic name
  geom_point(data = a,aes(x=coordinate_1,y=coordinate_2),cex=1) + 
  xlim (min(b$coordinate_1,a$coordinate_1)-0.2,max(b$coordinate_1,a$coordinate_1)+0.2) + 
  ylim(min(b$coordinate_2,a$coordinate_2)-0.2,max(b$coordinate_2,a$coordinate_2)+0.2)
  # xlim (-0.8,0.8) + ylim(-.8,.8) 
"""










######## distance

a["dist"] = (a["coordinate_1"] ** 2 + a["coordinate_2"] ** 2) ** 0.5
a_new = a.loc[(a["dist"] > 1.5), :] #TODO: 1.5 is right?




"""
pdf(paste0(dir,"/plot_wz_new.pdf"))
ggplot() + 
  geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = id),col=2) +
  # geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = topic_name),col=2) +  #topic name
  geom_point(data = a_new,aes(x=coordinate_1,y=coordinate_2),cex=1) + 
  xlim (min(b$coordinate_1,a_new$coordinate_1)-0.2,max(b$coordinate_1,a_new$coordinate_1)+0.2) + 
  ylim(min(b$coordinate_2,a_new$coordinate_2)-0.2,max(b$coordinate_2,a_new$coordinate_2)+0.2)
# xlim (-0.8,0.8) + ylim(-.8,.8) 

"""
output_new["beta_estimate"]

####################### TODO:: 3dplot

new = pd.DataFrame({"x": b["coordinate_1"],
                    "y": b["coordinate_2"],
                    "z": output_new["beta_estimate"],
                    "topics": b["topic_name"]})  
# word의 계수1, word의 계수2, beta의 측정치, word의 topic name,
# output_new는 z_est, w_est, z.proc, w.proc으로 바꾼버전

word_cluster = KMeans(n_clusters = 4).fit(output_new["z_estimate"]).labels_
topic_cluster = KMeans(n_clusters = 4).fit(b.iloc[:, 1:2]).labels_

new["colors"] = topic_cluster

#colors = topic_cluster


new

for subject in "Plot5_3dplot_beta":
    fig = px.scatter_3d(new, x = 'x', y = 'y', z = 'z', color = 'colors', title = subject,
                        text = "topics"
                        #text = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                        )
    fig.show()


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

#which.max(bet_tot)
#idist <- as.matrix(dist(x = as.data.frame(b[,1:2]), method = "euclidean"))


eps = np.finfo("double").eps #.Machine$double.eps

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



"""


which.max(bet_tot)

idist <- as.matrix(dist(x = as.data.frame(b[,1:2]), method = "euclidean"))
W = affinityMatrix(idist,K=which.max(bet_tot))
d = rowSums(W)
d[d == 0] = .Machine$double.eps #what?
D = diag(d)
L = D - W
Di = diag(1 / sqrt(d))
NL = Di %*% L %*% Di
eig = eigen(NL)


## cluster the topic using select number of cluster
res = sort(abs(eig$values),index.return = TRUE)
#ncluster <- min(k)
ncluster <- 3
U = eig$vectors[,res$ix[1:ncluster]]
normalize <- function(x) x / sqrt(sum(x^2))
U = t(apply(U,1,normalize))
final = kmeans(U, ncluster)
group <- final$cluster  

group_index <- c()
for(i in 1:max(group)){
  group_index[i] <- min(which(group==i))
}


"""






for a in "a":
    fig = px.scatter(b, x = "coordinate_1", y = "coordinate_2", 
                    text="topic_name", title = "Plot1_topic_cluter", color = "group")
    # geom_text(data = a[group_index,], aes(x=coordinate_1, y = coordinate_2+0.15, 
    # label = dbscan$cluster[group_index]),col=2)+
    #fig = px.scatter(b, x = "coordinate_1", y = "coordinate_2", 
    #                text="topic_name", title = "Plot1_topic_cluter", color = "group")
    fig.update_traces(textposition='top center')
    fig.show()





"""


pdf(paste0(dir,"/.pdf"))
##plot of topic 
ggg <- ggplot() +
  
  geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = topic_name),col=as.factor(group), cex=3) +
  # geom_point(data = a, aes(x=coordinate_1,y= coordinate_2), cex=1) +
  
  scale_color_manual(values = c("black", brewer.pal(n = 9, name = 'Set1'))) +
  # geom_point(data = a[!dbscan$isseed,],aes(coordinate_1, coordinate_2), shape=8, cex=1) +
  xlim (min(b$coordinate_1,a$coordinate_1)-0.2,max(b$coordinate_1,a$coordinate_1)+0.2) + 
  ylim(min(b$coordinate_2,a$coordinate_2)-0.2,max(b$coordinate_2,a$coordinate_2)+0.2) +
  #xlim (-0.8,1.2) + ylim(-.8,.1) + 
  theme_bw() + theme(legend.position = "None") 
print(ggg)
dev.off()


"""

###############################

word_position = pd.concat([data_word, a.iloc[:, 0:2]], axis=1)

word_position["dist"] = (word_position["coordinate_1"] ** 2 + word_position["coordinate_2"] ** 2) ** 0.5

word_new = word_position.loc[word_position["dist"] > 1.4, :]














for a in "a":
    fig = px.scatter(b, x = "coordinate_1", y = "coordinate_2", 
                    text="topic_name", title = "topic_cluter_with_words", color = "group")
    fig = px.scatter(a, x = "coordinate_1", y = "coordinate_2", 
                    text="topic_name", title = "topic_cluter_with_words", color = "group")



"""



pdf(paste0(dir,"/.pdf"))
##plot of topic 
ggg <- ggplot() +
  geom_point(data = b, aes(x=coordinate_1, y = coordinate_2),col=0.5) +
  geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = topic_name),col=as.factor(group), cex=3) +
  # geom_point(data = a, aes(x=coordinate_1,y= coordinate_2), cex=1) +
  # geom_text(data = a[group_index,], aes(x=coordinate_1, y = coordinate_2+0.15, label = dbscan$cluster[group_index]),col=2)+
  scale_color_manual(values = c("black", brewer.pal(n = 9, name = 'Set1'))) +
  # geom_point(data = a[!dbscan$isseed,],aes(coordinate_1, coordinate_2), shape=8, cex=1) +
  geom_point(data = a_new,aes(x=coordinate_1,y=coordinate_2),cex=1) + 
  geom_label_repel(data = word_new, aes(x=coordinate_1,y=coordinate_2, label = data_word))+
  xlim (min(b$coordinate_1,a_new$coordinate_1)-0.2,max(b$coordinate_1,a_new$coordinate_1)+0.2) + 
  ylim(min(b$coordinate_2,a_new$coordinate_2)-0.2,max(b$coordinate_2,a_new$coordinate_2)+0.2) +
  theme_bw() + theme(legend.position = "None") 
print(ggg)
dev.off()



"""





#quantile(word_position$dist, c(0.05, 0.2))
word_new = word_position[word_position["dist"] < 0.41,]

# word_new


from plotly.offline import iplot, init_notebook_mode
from plotly.subplots import make_subplots
init_notebook_mode()
import plotly.express as px

df = px.data.gapminder().query("year==2007 and continent=='Americas'")

fig = px.scatter(df, x="gdpPercap", y="lifeExp", text="country", log_x=True, size_max=60)

fig.update_traces(textposition='top center')

fig.update_layout(
    height=800,
    title_text='GDP and Life Expectancy (Americas, 2007)'
)

fig.show()



for subject in "Plot4_topic_cluter_centerwords":
    plt.title(subject)
    plt.scatter(data = b, x = "coordinate_1", y = "coordinate_2", colors = 0.5)
    plt.scatter(data = b, x = "coordinate_1", y = "coordinate_2", label = colnames(data_set2), col=as.factor(group), s = 240)
    plt.scatter(data = a, x = "coordinate_1", y = "coordinate_2", s = 80)
    plt.scatter(data = a[group_index,:], x = "coordinate_1", y = "coordinate_2" + 0.15, label = dbscan["cluster"][group_index], col = 2)
    #scale_color_manual(values = c("black", brewer.pal(n = 9, name = 'Set1')))
    #plt.scatter(data = a[!dbscan["isseed"],:], x = "coordinate_1", y = "coordinate_2", shape=8, cex=1)
    plt.scatter(data = a_new, x = "coordinate_1", y = "coordinate_2", cex=1)
    
    ### plt.xlim(np.min(b["coordinate_1"].min(), a_new["coordinate_1"].min()) - 0.2,
    ###          np.max(b["coordinate_1"].max(), a_new["coordinate_1"].max()) + 0.2)
    ### plt.ylim(np.min(b["coordinate_2"].min(), a_new["coordinate_2"].min()) - 0.2, 
    ###          np.max(b["coordinate_2"].max(), a_new["coordinate_2"].max()) + 0.2)
    
    theme_bw()
    theme(legend.position = "None") 
    continue






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


for subject in "Plot3_topic_close_words":
    close_word_index.append()
    "close_word_index[[i]] <- order(as.matrix(dist(rbind(b[i,1:2], a[,1:2])))[1,-1])[1:25]"
    
    
np.argsort()



dist(pd.concat([b.iloc[i, 1:2], a.iloc[:, 1:2]], axis = 1)).iloc[1,-1]

dist()


close_word_index[[i]] <- order(as.matrix(dist(rbind(b[i,1:2], a[,1:2])))[1,-1])[1:25]

"""
library(ggrepel)
pdf(paste0(dir,"Plot3_topic_close_words"))

for(i in 1:ncol(data_set2)){
  ## here 10
  #i = 1
  close_word_index[[i]] <- order(as.matrix(dist(rbind(b[i,1:2], a[,1:2])))[1,-1])[1:25]
  ggg <- ggplot() +
    geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = seq(1:ncol(data_set2))),col=as.factor(group), cex=3) +
    geom_point(data = word_position[close_word_index[[i]],], aes(x=coordinate_1, y = coordinate_2), cex=1) +
    geom_point(data = b[i,], aes(x=coordinate_1,y= coordinate_2), col='red', cex=7, shape=2, stroke = 1) +
    scale_color_manual(values = c("black", brewer.pal(n = 9, name = 'Set1'))) +
    xlim (min(b$coordinate_1,a$coordinate_1)-0.5,max(b$coordinate_1,a$coordinate_1)+0.5) + 
    ylim(min(b$coordinate_2,a$coordinate_2)-0.5,max(b$coordinate_2,a$coordinate_2)+0.5) + 
    labs(title=paste("Topic", i),
         x ="", y = "")+
    theme_bw() + theme(legend.position = "None") 
  
  # ggg <- ggg+ geom_text_repel(data = word_position[close_word_index,], aes(x=X1, y = X2, label = covid_20_word), size=3) 
  ggg1 <- ggg+ geom_label_repel(data = word_position[close_word_index[[i]],], aes(x=coordinate_1, y = coordinate_2, label = words),
                                # fontface = 'bold', color = 'white',
                                box.padding = unit(0.5, "lines"),
                                point.padding = unit(0.5, "lines"),
                                segment.color = 'grey50') 
  print(ggg1)
}
dev.off()


"""


##### Cluster_Group near words


def euc_dist(x1, x2): sqrt(sum((x1 - x2) ** 2))


# head(word_position)





for i in range(temp2.shape[0]):
    cluster_word_dist = []
    close_word_index = []

    for k in range(output_new["z_estimate"].shape[0]):
        cluster_word_dist.append(euc_dist(temp2[i,-1], output_new["z_estimate"][k,:]))
    np.argsort(np.array(cluster_word_dist))[1:25]

    for subject in "Plot2_cluster_close_words":
        plt.title(subject)
        plt.scatter(data = b, x = "coordinate_1", y = "coordinate_2")
        ### for i, txt in enumerate(df["id"]):
            ### plt.annotate(txt, (df["coordinate_1"].iat[i], df["coordinate_2"].iat[i]))
            ### plt.annotate(txt, (df.x.iat[i], df.y.iat[i]))
        plt.scatter(data = word_position[close_word_index, :], x = "coordinate_1", y = "coordinate_2", s = 80)

        plt.scatter(data = temp2[,-1], x = "x", y = "y",
                    label = paste("Group", seq(1,nrow(temp2)), sep = ""),
                    col = "red", s = 320
                    )

        plt.scatter(data = b[i,:],  x = "coordinate_1", y = "coordinate_2", col='red', cex=7, shape=2, stroke = 1)
        """
        scale_color_manual(values = c("black", brewer.pal(n = 9, name = 'Set1')))
        labs(title=paste("Cluster", i), x ="", y = "")
        theme_bw() + theme(legend.position = "None") 
        """
    
    for subject in "ggg1":
        """
        geom_label_repel(data = word_position[close_word_index,], 
                        aes(x=coordinate_1, y = coordinate_2, label = words),
                        # fontface = 'bold', color = 'white',
                        box.padding = unit(0.5, "lines"),
                        point.padding = unit(0.5, "lines"),
                        segment.color = 'grey50') 
        """






"""

pdf(paste0(dir,"/Plot2_cluster_close_words.pdf"))
for(i in 1:nrow(temp2)){
  ggg <- ggplot() +
    
    
    scale_color_manual(values = c("black", brewer.pal(n = 9, name = 'Set1'))) +
    labs(title=paste("Cluster", i), x ="", y = "")+
    theme_bw() + theme(legend.position = "None") 
  print(ggg)

  ggg1 <- ggg + 
  
  print(ggg1)
}

dev.off()

save.image("75_result.RData")



"""



#save.image("75_result.RData")

