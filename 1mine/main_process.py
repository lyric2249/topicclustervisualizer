from math import *

import numpy as np
import pandas as pd

#from pkgp.functionp import *
from src.functionp import *

#functionp.functionp

#from ../functionp/functionp import *
#import function1.lsrm1pl_normal_missing_mine

import scipy


data = pd.DataFrame()

data_word = data.iloc[:, 0]
data_source = data.columns[1:]
data = data.drop["words"]
data = data.fillna(99)

data_m = np.array(data)
#logit_x = log(data_m / (1 - data_m))
data_m = log(data_m / (1 - data_m))

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


# Set 99 as missing
output = onepl_lsrm_cont_missing(data,

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

                                 99)

(
    """
    #output["accept_beta"].T
    #output["accept_theta"]
    ### View(t(output$accept_theta))

    #output["accept_w"].T
    #output["accept_gamma"].T
    ### View(t(output$accept_z))

    #output$accept_z
    ##Rcode procrustes matching
    """  
) #TODO:: View Output Table?






################################################################################################

def MCMC_process(data):
    nsample = data.shape[0]
    nitem = data.shape[1]

    nmcmc = int((niter - nburn) / nthin)

    max_address = min(output['map'].index(max(output['map'])))

    w_star = output['w'].iloc[max_address, :, :]
    z_star = output['z'].iloc[max_address, :, :]

    w_proc = np.zeros((nmcmc, nitem, ndim), )
    z_proc = np.zeros((nmcmc, nsample, ndim), )

    for iter in range(nmcmc):
        z_iter = output['z'].iloc[iter, :, :]

        if iter != max_address:
            z_proc[iter, :, :] = procrustes_mine(z_iter, z_star)
            # z_proc[iter,:,:] = """procrustes(z_iter, z_star)$X.new""" #TODO: ======= 210717 =======
        else:
            z_proc[iter, :, :] = z_iter

        w_iter = output['w'].iloc[iter, :, :]

        if iter != max_address:
            w_proc[iter, :, :] = procrustes_mine(w_iter, w_star)  # TODO: ======= 210717 =======
        else:
            w_proc[iter, :, :] = w_iter

    w_est = np.empty((nitem, ndim,))

    for i in range(nitem):
        for j in range(ndim):
            w_est[i, j] = w_proc[:, i, j].mean

    z_est = np.empty((nsample, ndim,))

    for k in range(nsample):
        for j in range(ndim):
            z_est[k, j] = z_proc[:, k, j].mean

    beta_est = output["beta"].mean
    theta_est = output["theta"].mean

    # beta_est = apply(output["beta"], 2, mean)
    # theta_est = apply(output["theta"], 2, mean)

    sigma_theta_est = output["sigma_theta"].mean
    gamma_est = output["gamma"].mean

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

################################################################################################

output_new = MCMC_process(data)

"""
#t(output_new$accept_beta)
#t(output_new$accept_theta)
#t(output_new$accept_w)
#t(output_new$accept_z)

#output_new["accept_gamma"]

# save.image("full_continuous_60.RData")
# save.image("full_continuous_70.RData")
# save.image("full_continuous_80.RData")
# save.image("full_continuous_mean.RData")
"""

"""
save.image("90_result.RData")

dir
pdf(paste0(dir, "/trace_beta.pdf"))
for (i in 1:ncol(output_new$beta)) ts.plot(output_new$beta[1:nrow(output_new$beta), i], main = paste("beta", i))
dev.off()

pdf(paste0(dir, "/trace_theta.pdf"))
for (i in 1:ncol(output_new$theta)) ts.plot(output_new$theta[, i], main=paste("theta", i))
dev.off()

pdf(paste0(dir, "/trace_sigma_theta.pdf"))
ts.plot(output_new$theta_sd, main = "sigma_theta")
dev.off()

pdf(paste0(dir, "/trace_gamma.pdf"))
ts.plot(output_new$gamma, main = "gamma")
dev.off()
"""
# </editor-fold>


"""
pdf(paste0(dir, "/trace_z.pdf"))
for (k in 1:ncol(output_new$theta)) ts.plot(output_new$z[, k, 1], main=paste("z_1", k))
for (k in 1:ncol(output_new$theta)) ts.plot(output_new$z[, k, 2], main=paste("z_2", k))
dev.off()

pdf(paste0(dir, "/trace_w.pdf"))
for (i in 1:ncol(output_new$beta)) ts.plot(output_new$w[, i, 1], main=paste("w_1", i))
for (i in 1:ncol(output_new$beta)) ts.plot(output_new$w[, i, 2], main=paste("w_2", i))
dev.off()
"""  # TODO: Multiple Write multiple Time Series Plot in pdf

## lsrm plot
#### ggplot

a = pd.DataFrame(output_new["z_estimate"], columns=["coordinate_1", "coordinate_2"])
b = pd.DataFrame(output_new["w_estimate"], columns=["coordinate_1", "coordinate_2"])  # x와 w의 coordinate로 df 구성

b["topic_name"] = data_m.columns
b["id"] = range(1, data_m.shape[1] + 1)

############### Rotate


angle = -pi / 30

M = pd.DataFrame([[cos(angle), sin(angle)], [-sin(angle), cos(angle)]], columns=["coordinate_1", "coordinate_2"])
# clockwise 회전용도. 따라서 일반적인 회전행렬의 inv. clockwise 기준 -6도씩 돌림.


bnew = (-b[:, 1:2]).dot(M)  # TODO: =========================================== #
# bnew = data.frame(as.matrix(-b[, 1:2]) % * % M)
# bnew = data.frame(as.matrix(-b[,2:1]) %*% M) # for 75, 70 # 계수 컬럼1과 컬럼2가 교환됨. for what?


bnew["topic_name"] = b["topic_name"]
bnew["id"] = b["id"]

bnew.columns = b.columns

# head(bnew)
# head(a)

anew = (-a[:, 1:2]).dot(M)  # TODO: ===========================================
# anew = (-a[:, 2:1]).dot(M)

anew.columns = ["coordinate_1", "coordinate_2"]

"""

gg = ggplot() +
    geom_text(data=bnew, aes(x=coordinate_1, y=coordinate_2, label=id), col=2) +
    # geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = topic_name),col=2) +  #topic name
    # geom_point(data = anew,aes(x=coordinate_1,y=coordinate_2),cex=1) +
    xlim(min(bnew$coordinate_1, anew$coordinate_1)-0.2, max(bnew$coordinate_1, anew$coordinate_1)+0.2) +
    ylim(min(bnew$coordinate_2, anew$coordinate_2)-0.2, max(bnew$coordinate_2, anew$coordinate_2)+0.2)
    # xlim (-0.8,0.8) + ylim(-.8,.8)
print(gg)

b = bnew
a = anew

dir





pdf(paste0(dir, "/plot_wz.pdf"))



gg = ggplot() +
    geom_text(data=b, aes(x=coordinate_1, y=coordinate_2, label=id), col=2) +
    # geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = topic_name),col=2) +  #topic name
    geom_point(data=a, aes(x=coordinate_1, y=coordinate_2), cex=1) +
    xlim(min(b$coordinate_1, a$coordinate_1)-0.2, max(b$coordinate_1, a$coordinate_1)+0.2) +
    ylim(min(b$coordinate_2, a$coordinate_2)-0.2, max(b$coordinate_2, a$coordinate_2)+0.2)
    # xlim (-0.8,0.8) + ylim(-.8,.8)
print(gg)

gg = ggplot() +
    # geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = id),col=2) +
    geom_text(data=b, aes(x=coordinate_1, y=coordinate_2, label=topic_name), col=2) +  # topic name
    geom_point(data=a, aes(x=coordinate_1, y=coordinate_2), cex=1) +
    xlim(min(b$coordinate_1, a$coordinate_1)-0.2, max(b$coordinate_1, a$coordinate_1)+0.2) +
    ylim(min(b$coordinate_2, a$coordinate_2)-0.2, max(b$coordinate_2, a$coordinate_2)+0.2)
print(gg)



dev.off()
"""

################################################## TODO: distance


a["dist"] = (a["coordinate_1"] ** 2 + a["coordinate_2"] ** 2) ** 0.5

# head(a)
# hist(a$dist)
# summary(a$dist)
# quantile(a$dist, c(0.8, 0.9))

a_new = a[a["dist"] > 1.5, :]

"""
pdf(paste0(dir, "/plot_wz_new.pdf"))



gg = ggplot() +
    geom_text(data=b, aes(x=coordinate_1, y=coordinate_2, label=id), col=2) +
    # geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = topic_name),col=2) +  #topic name
    geom_point(data=a_new, aes(x=coordinate_1, y=coordinate_2), cex=1) +
    xlim(min(b$coordinate_1, a_new$coordinate_1)-0.2, max(b$coordinate_1, a_new$coordinate_1)+0.2) +
    ylim(min(b$coordinate_2, a_new$coordinate_2)-0.2, max(b$coordinate_2, a_new$coordinate_2)+0.2)
    # xlim (-0.8,0.8) + ylim(-.8,.8)
print(gg)

gg = ggplot() +
    # geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = id),col=2) +
    geom_text(data=b, aes(x=coordinate_1, y=coordinate_2, label=topic_name), col=2) +  # topic name
    geom_point(data=a_new, aes(x=coordinate_1, y=coordinate_2), cex=1) +
    xlim(min(b$coordinate_1, a_new$coordinate_1)-0.2, max(b$coordinate_1, a_new$coordinate_1)+0.2) +
    ylim(min(b$coordinate_2, a_new$coordinate_2)-0.2, max(b$coordinate_2, a_new$coordinate_2)+0.2)
print(gg)



dev.off()
"""

####################### TODO:: 3dplot

# par(mfrow=c(1, 1))
# library(scatterplot3d)

new = pd.DataFrame({"x": b["coordinate_1"],
                    "y": b["coordinate_2"],
                    "z": output_new["beta_estimate"],
                    "topics": b["topic_name"]})  # word의 계수1, word의 계수2, beta의 측정치, word의 topic name,
# output_new는 z_est, w_est, z.proc, w.proc으로 바꾼버전



word_cluster = kmeans(output_new["z_estimate"], 4)
# word_cluster["cluster"]

topic_cluster = kmeans(b.iloc[:, 1:2], 4)
# topic_cluster["cluster"]

colors = topic_cluster["cluster"]

topic_plot = scatterplot3d(new[:, 1:3],
                           pch=16,
                           color=colors,
                           angle=50)  # TODO: =================================
# word_plot$points3d(output_new$w_estimate,pch=8,color=color2)


"""
pdf(paste0(dir, "/Plot5_3dplot_beta.pdf"))



topic_plot = scatterplot3d(new[, -4], pch = 16, color = colors, angle = 50)
text(topic_plot$xyz.convert(new[, -4]+0.3), labels = new$topics)

topic_plot = scatterplot3d(new[, -4], pch = 16, color = colors, angle = 50)
text(topic_plot$xyz.convert(new[, -4]+0.3), labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                                       19, 20))


dev.off()
"""
####################################

wcss = bet_tot = bet = [1, 2]
ncluster = 5

data_set2 = data_m

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from scipy.spatial.distance import euclidean

for aa in range(3, (ncol(data_set2))):
    # aa=3

    X = b.iloc[:, 1:2]

    idist = np.zeros((X.shape[1], X.shape[1]))

    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            idist[i, j] = euclidean(X[:, i], X[:, j])
            continue


    # idist = as.matrix(dist(x= as.data.frame(b.iloc[:, 1:2]), method = "euclidean"))  # TODO: ====================

    
    W = affinityMatrix(idist, K=aa)  # TODO: =====================
    d = W.sum(axis=1)

    d[d == 0] = np.finfo("double").eps

    D = np.diag(d)

    L = D - W
    Di = np.diag(1 / sqrt(d))

    NL = Di @ L @ Di
    ev = np.linalg.eig(NL)[0]
    evec = np.linalg.eig(NL)[1]

    ix = pd.Index(abs(ev)).sort_values(return_indexer=True)[1]
    U = evec[:, ix[1:ncluster]]


    def normalize(x):
        return x / sqrt(sum(x ^ 2))


    U.apply(normalize, 1)

    final = KMeans(n_clusters=ncluster, random_state=0).fit(U)
    group = final.labels_

    totss = sum(sum(scale(X, axis=0, with_std=False) ** 2))
    tot_withinss = final.inertia_
    betweenss = totss - tot_withinss

    wcss[aa] = tot_withinss  # min #3:82%, 4: 61%, 5: 91%, 6:86%. 7:83%. 8: 85%, 9: 86%. 10: 87%
    bet_tot[aa] = betweenss / totss * 100  # max
    bet[aa] = betweenss

    # wcss[aa] < - final$tot.withinss  # min #3:82%, 4: 61%, 5: 91%, 6:86%. 7:83%. 8: 85%, 9: 86%. 10: 87%
    # bet_tot[aa] < - final$betweenss / final$totss * 100  # max
    # bet[aa] < - final$betweenss

# which.max(bet_tot)


# TODO: =====================


W = affinityMatrix(idist, K = np.argmax(bet_tot)) # TODO: argmax 반환값이 어레이면 오류발생 가능성 # TODO: 

#W = affinityMatrix(idist, K=which.max(bet_tot))  # TODO: =====================

## cluster the topic using select number of cluster
# ncluster <- min(k)  # TODO: =====================
ncluster = 3  # TODO: =====================


group_index = []

for i in range(1, max(group) + 1):
    group_index[i] = min(np.where(group == i))

# TODO: =====================

"""
##plot of topic

pdf(paste0(dir, "/Plot1_topic_cluter.pdf")) 



ggg = ggplot() +
    geom_point(data=b, aes(x=coordinate_1, y=coordinate_2), col=0.5) +
    geom_text(data=b, aes(x=coordinate_1, y=coordinate_2, label=topic_name), col=as.factor(group), cex = 3) +
    # geom_point(data = a, aes(x=coordinate_1,y= coordinate_2), cex=1) +
    # geom_text(data = a[group_index,], aes(x=coordinate_1, y = coordinate_2+0.15, label = dbscan$cluster[group_index]),col=2) +
    scale_color_manual(values=c("black", brewer.pal(n=9, name='Set1'))) +
    # geom_point(data = a[!dbscan$isseed,],aes(coordinate_1, coordinate_2), shape=8, cex=1) +
    xlim(min(b$coordinate_1, a$coordinate_1)-0.2, max(b$coordinate_1, a$coordinate_1)+0.2) +
    ylim(min(b$coordinate_2, a$coordinate_2)-0.2, max(b$coordinate_2, a$coordinate_2)+0.2) +
    # xlim (-0.8,1.2) + ylim(-.8,.1) +
    theme_bw() + theme(legend.position = "None")

print(ggg)



dev.off()
"""

###############################

word_position = pd.concat([data_word, a.iloc[:, 0:1]], axis=1)

word_position["dist"] = (word_position["coordinate_1"] ** 2 + word_position["coordinate_2"] ** 2) ** 0.5
word_new = word_position.loc[word_position["dist"] > 1.4, :]

"""
##plot of topic

pdf(paste0(dir, "/topic_cluter_withwords.pdf"))


ggg = ggplot() +
    geom_point(data=b, aes(x=coordinate_1, y=coordinate_2), col=0.5) +
    geom_text(data=b, aes(x=coordinate_1, y=coordinate_2, label=topic_name), col=as.factor(group), cex = 3) +
    # geom_point(data = a, aes(x=coordinate_1,y= coordinate_2), cex=1) +
    # geom_text(data = a[group_index,], aes(x=coordinate_1, y = coordinate_2+0.15, label = dbscan$cluster[group_index]),col=2)+
    scale_color_manual(values=c("black", brewer.pal(n=9, name='Set1'))) +
    # geom_point(data = a[!dbscan$isseed,],aes(coordinate_1, coordinate_2), shape=8, cex=1) +
    geom_point(data=a_new, aes(x=coordinate_1, y=coordinate_2), cex=1) +
    geom_label_repel(data=word_new, aes(x=coordinate_1, y=coordinate_2, label=data_word)) +
    xlim(min(b$coordinate_1, a_new$coordinate_1)-0.2, max(b$coordinate_1, a_new$coordinate_1)+0.2) +
    ylim(min(b$coordinate_2, a_new$coordinate_2)-0.2, max(b$coordinate_2, a_new$coordinate_2)+0.2) +
    theme_bw() + theme(legend.position = "None")
print(ggg)



dev.off()
"""

#quantile(word_position$dist, c(0.05, 0.2))
word_new = word_position[word_position["dist"] < 0.41,]

# word_new

"""
pdf(paste0(dir,"/Plot4_topic_cluter_centerwords.pdf"))

##plot of topic 
ggg = ggplot() +
  geom_point(data = b, aes(x=coordinate_1, y = coordinate_2),col=0.5) +
  geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = colnames(data_set2)),col=as.factor(group), cex=3) +
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

words = data_word
close_word_index = []

## cosine similarity

temp = pd.concat([b.iloc[:, 1:2], group], axis=1)
# head(temp)

## center of topic group


# temp2 = data.frame(temp % > % group_by(group) % > % summarise(x=mean(coordinate_1), y=mean(coordinate_2)))

temp2 = temp.groupby("group", axis=1).agg({"coordinate_1": "mean", "coordinate_2": "mean"})
temp2.columns = ["x", "y"]

# head(temp2)

word_position = pd.concat([words, a[:, 1:2]], axis=1)
close_word_index = []
# head(word_position)

#library(ggrepel)

"""
pdf(paste0(dir, "/Plot3_topic_close_words.pdf"))

for (i in 1:ncol(data_set2)){
    ## here 10
    # i = 1
    close_word_index[[i]] = order(as.matrix(dist(rbind(b[i, 1:2], a[, 1:2])))[1, -1])[1: 25]
    ggg = ggplot() +
        geom_text(data=b, aes(x=coordinate_1, y=coordinate_2, label=seq(1: ncol(data_set2))), col =as.factor(group), cex = 3) +
        geom_point(data=word_position[close_word_index[[i]],], aes(x=coordinate_1, y=coordinate_2), cex=1) +
        geom_point(data=b[i,], aes(x=coordinate_1, y=coordinate_2), col='red', cex=7, shape=2, stroke=1) +
        scale_color_manual(values=c("black", brewer.pal(n=9, name='Set1'))) +
        xlim(min(b$coordinate_1, a$coordinate_1)-0.5, max(b$coordinate_1, a$coordinate_1)+0.5) +
        ylim(min(b$coordinate_2, a$coordinate_2)-0.5, max(b$coordinate_2, a$coordinate_2)+0.5) +
        labs(title=paste("Topic", i), x="", y="") +
        theme_bw() + theme(legend.position = "None")

    # ggg <- ggg+ geom_text_repel(data = word_position[close_word_index,], aes(x=X1, y = X2, label = covid_20_word), size=3)

    ggg1 = ggg + 
        geom_label_repel(data=word_position[close_word_index[[i]],],
                         aes(x=coordinate_1, y=coordinate_2, label=words),
                         # fontface = 'bold', color = 'white', box.padding = unit(0.5, "lines"),
                         point.padding = unit(0.5, "lines"),
                         segment.color = 'grey50'
                        )
    print(ggg1)
}
dev.off()
"""

##### Cluster_Group near words


def euc_dist(x1, x2): sqrt(sum((x1 - x2) ** 2))


# head(word_position)

"""
pdf(paste0(dir, "/Plot2_cluster_close_words.pdf"))



for (i in 1:nrow(temp2)){
    # i = 1
    ## here 10
    cluster_word_dist < -c()
    close_word_index < -c()

    for (k in c(1:nrow(output_new$z_estimate))){
        cluster_word_dist[k]=euc.dist(temp2[i, -1], output_new$z_estimate[k, ])
    }
    close_word_index < -order(as.matrix(cluster_word_dist))[1: 25]
    ggg = ggplot() +
        geom_text(data=b, aes(x=coordinate_1, y=coordinate_2, label=seq(1: ncol(data_set2))), col =as.factor(group), cex = 3) +
        geom_point(data=word_position[close_word_index,], aes(x=coordinate_1, y=coordinate_2), cex=1) +
        geom_text(data=temp2[, -1], aes(x=x, y=y, label=paste("Group", seq(1, nrow(temp2)), sep="")), col = "red", cex = 3) +
        # geom_point(data = b[i,], aes(x=coordinate_1,y= coordinate_2), col='red', cex=7, shape=2, stroke = 1) +
        scale_color_manual(values=c("black", brewer.pal(n=9, name='Set1'))) +
        labs(title=paste("Cluster", i), x="", y="") +
        theme_bw() + theme(legend.position = "None")

    print(ggg)
    ggg1 = ggg + geom_label_repel(data=word_position[close_word_index,], 
                                    aes(x=coordinate_1, y=coordinate_2, label=words),
                                    # fontface = 'bold', color = 'white',
                                    box.padding = unit(0.5, "lines"),
                                    point.padding = unit(0.5, "lines"),
                                    segment.color = 'grey50'
                                )
    print(ggg1)
}



dev.off()
"""

#save.image("75_result.RData")

