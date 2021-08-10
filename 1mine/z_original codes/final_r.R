library(readxl)
library(rstudioapi)
library(lsrm12pl)
library("SNFtool")
library(flexclust)
library("fpc")
library("ggplot2")
# library(factoextra)
library(RColorBrewer)
library(dbscan)
library(readxl)
library(readr)
library(Rcpp)
library(MCMCpack)
library(dplyr)
library(ggrepel)

113-40

current_path <-rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path ))
getwd()


sourceCpp("lsrm1pl_normal_missing.cpp")


dir = "/Users/yeseul/Dropbox/Kelly/대학원/진익훈교수님/gViz/90_result"
dir = "/Users/jeon-yeseul/Dropbox/Kelly/대학원/진익훈교수님/gViz/75_result"


data<-read.csv("./data_prob_90.csv",fileEncoding = guess_encoding("./data_prob_90.csv")[1,1] %>% as.character)
head(data)




data_word <- data[,1]
data<-data[,-1]
data[is.na(data)] <- 99
data_m = as.matrix(data)
logit.x = log(data_m /(1-data_m ))
head(logit.x)

### covid_20
ndim     <- 2 
niter    <- 55000
nburn    <- 5000
nthin    <- 5
nprint   <- 5000
jump_beta     <- 0.3
jump_theta    <- 1.0
jump_w        <- 0.06
jump_z        <- 0.5
jump_gamma    <- 0.01


pr_mean_beta  <- 0
pr_sd_beta    <- 1
pr_mean_theta <- 0
pr_sd_theta   <- 1
pr_mean_gamma <- 0.0
pr_sd_gamma <- 1.0
pr_a_sigma <- 0.001
pr_b_sigma <- 0.001
pr_a_th_sigma <- 0.001
pr_b_th_sigma <- 0.001


#Set 99 as missing 
output <- onepl_lsrm_cont_missing(as.matrix(data),
                                  ndim, niter, nburn, nthin, nprint,
                                  jump_beta, jump_theta,  jump_gamma, jump_z, jump_w,
                                  pr_mean_beta, pr_sd_beta, pr_a_th_sigma, pr_b_th_sigma, pr_mean_theta,
                                  pr_a_sigma, pr_b_sigma,pr_mean_gamma, pr_sd_gamma, 99)


output <- onepl_lsrm_cont_missing(as.matrix(logit.x),
                                  ndim, niter, nburn, nthin, nprint,
                                  jump_beta, jump_theta,  jump_gamma, jump_z, jump_w,
                                  pr_mean_beta, pr_sd_beta, pr_a_th_sigma, pr_b_th_sigma, pr_mean_theta,
                                  pr_a_sigma, pr_b_sigma,pr_mean_gamma, pr_sd_gamma, 99)


t(output$accept_beta)
output$accept_theta 
# View(t(output$accept_theta))
t(output$accept_w)
t(output$accept_gamma)
# View(t(output$accept_z))
output$accept_z
##Rcode procurst matching


{  nsample <- nrow(data)
  nitem <- ncol(data)
  
  nmcmc = as.integer((niter - nburn) / nthin)
  max.address = min(which.max(output$map))
  w.star = output$w[max.address,,]
  z.star = output$z[max.address,,]
  w.proc = array(0,dim=c(nmcmc,nitem,ndim))
  z.proc = array(0,dim=c(nmcmc,nsample,ndim))
  
  for(iter in 1:nmcmc){
    z.iter = output$z[iter,,]
    if(iter != max.address) z.proc[iter,,] = procrustes(z.iter,z.star)$X.new
    else z.proc[iter,,] = z.iter
    
    w.iter = output$w[iter,,]
    if(iter != max.address) w.proc[iter,,] = procrustes(w.iter,w.star)$X.new
    else w.proc[iter,,] = w.iter
  }
  
  w.est = matrix(NA,nitem,ndim)
  for(i in 1:nitem){
    for(j in 1:ndim){
      w.est[i,j] = mean(w.proc[,i,j])
    }
  }
  z.est = matrix(NA,nsample,ndim)
  for(k in 1:nsample){
    for(j in 1:ndim){
      z.est[k,j] = mean(z.proc[,k,j])
    }
  }
  
  beta.estimate = apply(output$beta, 2, mean)
  theta.estimate = apply(output$theta, 2, mean)
  sigma_theta.estimate = mean(output$sigma_theta)
  gamma.estimate = mean(output$gamma)
  
  output_new <- list(beta_estimate  = beta.estimate,
                     theta_estimate = theta.estimate,
                     sigma_theta_estimate    = sigma_theta.estimate,
                     gamma_estimate    = gamma.estimate,
                     z_estimate     = z.est,
                     w_estimate     = w.est,
                     beta           = output$beta,
                     theta          = output$theta,
                     theta_sd       = output$sigma_theta,
                     gamma          = output$gamma,
                     z              = z.proc,
                     w              = w.proc,
                     accept_beta    = output$accept_beta,
                     accept_theta   = output$accept_theta,
                     accept_w       = output$accept_w,
                     accept_z       = output$accept_z,
                     accept_gamma   = output$accept_gamma)
  
}

t(output_new$accept_beta)
t(output_new$accept_theta)
t(output_new$accept_w)
t(output_new$accept_z)
output_new$accept_gamma
# save.image("full_continuous_60.RData")
# save.image("full_continuous_70.RData")
# save.image("full_continuous_80.RData")
# save.image("full_continuous_mean.RData")

save.image("90_result.RData")



dir
pdf(paste0(dir,"/trace_beta.pdf"))
for(i in 1:ncol(output_new$beta)) ts.plot(output_new$beta[1:nrow(output_new$beta),i],main=paste("beta",i))
dev.off()

pdf(paste0(dir,"/trace_theta.pdf"))
for(i in 1:ncol(output_new$theta)) ts.plot(output_new$theta[,i],main=paste("theta",i))
dev.off()  


pdf(paste0(dir,"/trace_sigma_theta.pdf"))
ts.plot(output_new$theta_sd,main="sigma_theta")
dev.off()

pdf(paste0(dir,"/trace_gamma.pdf"))
ts.plot(output_new$gamma,main="gamma")
dev.off()


#########################################################
pdf(paste0(dir,"/trace_z.pdf"))
for(k in 1:ncol(output_new$theta)) ts.plot(output_new$z[,k,1],main=paste("z_1",k))
for(k in 1:ncol(output_new$theta)) ts.plot(output_new$z[,k,2],main=paste("z_2",k))
dev.off()

pdf(paste0(dir,"/trace_w.pdf"))
for(i in 1:ncol(output_new$beta)) ts.plot(output_new$w[,i,1],main=paste("w_1",i))
for(i in 1:ncol(output_new$beta)) ts.plot(output_new$w[,i,2],main=paste("w_2",i))
dev.off()

## lsrm plot
#### ggplot
a <- as.data.frame(output_new$z_estimate)
colnames(a) <- c("coordinate_1","coordinate_2")
b <- as.data.frame(output_new$w_estimate)
colnames(b) <- c("coordinate_1","coordinate_2")
b$topic_name <- colnames(data_m) 
b$id <- 1:ncol(data_m)




############### Rotate
angle <- -pi/30
M <- matrix( c(cos(angle), -sin(angle), sin(angle), cos(angle)), 2, 2 )
bnew=data.frame(as.matrix(-b[,1:2]) %*% M)
#bnew=data.frame(as.matrix(-b[,2:1]) %*% M) # for 75, 70 
head(bnew)
bnew$topic_name = b$topic_name
bnew$id = b$id
colnames(bnew) = colnames(b)
head(bnew)
head(a)

anew=data.frame(as.matrix(-a[,1:2]) %*% M)
anew=data.frame(as.matrix(-a[,2:1]) %*% M) # for 75

colnames(anew) <- c("coordinate_1","coordinate_2")

gg <- ggplot() + 
  geom_text(data = bnew, aes(x=coordinate_1, y = coordinate_2, label = id),col=2) +
  # geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = topic_name),col=2) +  #topic name
  #geom_point(data = anew,aes(x=coordinate_1,y=coordinate_2),cex=1) + 
  xlim (min(bnew$coordinate_1,anew$coordinate_1)-0.2,max(bnew$coordinate_1,anew$coordinate_1)+0.2) + 
  ylim(min(bnew$coordinate_2,anew$coordinate_2)-0.2,max(bnew$coordinate_2,anew$coordinate_2)+0.2)
# xlim (-0.8,0.8) + ylim(-.8,.8) 
print(gg)

b = bnew
a = anew

dir
pdf(paste0(dir,"/plot_wz.pdf"))
gg <- ggplot() + 
  geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = id),col=2) +
  # geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = topic_name),col=2) +  #topic name
  geom_point(data = a,aes(x=coordinate_1,y=coordinate_2),cex=1) + 
  xlim (min(b$coordinate_1,a$coordinate_1)-0.2,max(b$coordinate_1,a$coordinate_1)+0.2) + 
  ylim(min(b$coordinate_2,a$coordinate_2)-0.2,max(b$coordinate_2,a$coordinate_2)+0.2)
# xlim (-0.8,0.8) + ylim(-.8,.8) 
print(gg)

gg <- ggplot() + 
  # geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = id),col=2) +
  geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = topic_name),col=2) +  #topic name
  geom_point(data = a,aes(x=coordinate_1,y=coordinate_2),cex=1) + 
  xlim (min(b$coordinate_1,a$coordinate_1)-0.2,max(b$coordinate_1,a$coordinate_1)+0.2) + 
  ylim(min(b$coordinate_2,a$coordinate_2)-0.2,max(b$coordinate_2,a$coordinate_2)+0.2)
print(gg)
dev.off()


################################################## distance 
a$dist = (a$coordinate_1^2+a$coordinate_2^2)^0.5 

head(a)
hist(a$dist)
summary(a$dist)
quantile(a$dist,c(0.8,0.9))

a_new = a[a$dist>1.5,]

pdf(paste0(dir,"/plot_wz_new.pdf"))
gg <- ggplot() + 
  geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = id),col=2) +
  # geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = topic_name),col=2) +  #topic name
  geom_point(data = a_new,aes(x=coordinate_1,y=coordinate_2),cex=1) + 
  xlim (min(b$coordinate_1,a_new$coordinate_1)-0.2,max(b$coordinate_1,a_new$coordinate_1)+0.2) + 
  ylim(min(b$coordinate_2,a_new$coordinate_2)-0.2,max(b$coordinate_2,a_new$coordinate_2)+0.2)
# xlim (-0.8,0.8) + ylim(-.8,.8) 
print(gg)

gg <- ggplot() + 
  # geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = id),col=2) +
  geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = topic_name),col=2) +  #topic name
  geom_point(data = a_new,aes(x=coordinate_1,y=coordinate_2),cex=1) + 
  xlim (min(b$coordinate_1,a_new$coordinate_1)-0.2,max(b$coordinate_1,a_new$coordinate_1)+0.2) + 
  ylim(min(b$coordinate_2,a_new$coordinate_2)-0.2,max(b$coordinate_2,a_new$coordinate_2)+0.2)
print(gg)
dev.off()


####################### 3dplot #####
####################################
par(mfrow=c(1,1))
library(scatterplot3d)
new = data.frame(x = b$coordinate_1, y =b$coordinate_2 , z = output_new$beta_estimate, topics = b$topic_name)

word_cluster=kmeans(output_new$z_estimate,4)
word_cluster$cluster

topic_cluster=kmeans(b[,1:2],4)
topic_cluster$cluster
colors = topic_cluster$cluster 
topic_plot=scatterplot3d(new[,1:3],pch=16,color = colors,angle = 50)
#word_plot$points3d(output_new$w_estimate,pch=8,color=color2)

pdf(paste0(dir,"/Plot5_3dplot_beta.pdf"))
topic_plot=scatterplot3d(new[,-4],pch=16,color = colors,angle = 50)
text(topic_plot$xyz.convert(new[,-4]+0.3),labels=new$topics)

topic_plot=scatterplot3d(new[,-4],pch=16,color = colors,angle = 50)
text(topic_plot$xyz.convert(new[,-4]+0.3),labels=c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20))
dev.off()
####################################


wcss <- c(1, 2)
bet_tot <- c(1, 2)
bet <- c(1, 2)
ncluster = 5
data_set2<-data_m
for(aa in 3:(ncol(data_set2)-1)){
  #aa=3
  idist <- as.matrix(dist(x = as.data.frame(b[,1:2]), method = "euclidean"))
  W = affinityMatrix(idist,K=aa)
  d = rowSums(W)
  d[d == 0] = .Machine$double.eps 
  D = diag(d)
  L = D - W
  Di = diag(1 / sqrt(d))
  NL = Di %*% L %*% Di
  eig = eigen(NL)
  
  res = sort(abs(eig$values),index.return = TRUE)
  U = eig$vectors[,res$ix[1:ncluster]]
  
  normalize <- function(x) x / sqrt(sum(x^2))
  U = t(apply(U,1,normalize))
  final = kmeans(U, ncluster)
  group <- final$cluster
  
  wcss[aa] <- final$tot.withinss #min #3:82%, 4: 61%, 5: 91%, 6:86%. 7:83%. 8: 85%, 9: 86%. 10: 87%
  bet_tot[aa] <- final$betweenss/final$totss*100 #max
  bet[aa]<- final$betweenss
}

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

pdf(paste0(dir,"/Plot1_topic_cluter.pdf"))
##plot of topic 
ggg <- ggplot() +
  geom_point(data = b, aes(x=coordinate_1, y = coordinate_2),col=0.5) +
  geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = topic_name),col=as.factor(group), cex=3) +
  # geom_point(data = a, aes(x=coordinate_1,y= coordinate_2), cex=1) +
  # geom_text(data = a[group_index,], aes(x=coordinate_1, y = coordinate_2+0.15, label = dbscan$cluster[group_index]),col=2)+
  scale_color_manual(values = c("black", brewer.pal(n = 9, name = 'Set1'))) +
  # geom_point(data = a[!dbscan$isseed,],aes(coordinate_1, coordinate_2), shape=8, cex=1) +
  xlim (min(b$coordinate_1,a$coordinate_1)-0.2,max(b$coordinate_1,a$coordinate_1)+0.2) + 
  ylim(min(b$coordinate_2,a$coordinate_2)-0.2,max(b$coordinate_2,a$coordinate_2)+0.2) +
  #xlim (-0.8,1.2) + ylim(-.8,.1) + 
  theme_bw() + theme(legend.position = "None") 
print(ggg)
dev.off()



###############################

word_position <- data.frame(data_word,a[,1:2])
word_position
head(word_position)

word_position$dist = (word_position$coordinate_1^2+word_position$coordinate_2^2)^0.5
head(word_position)

word_new = word_position[word_position$dist>1.4,]
word_new


pdf(paste0(dir,"/topic_cluter_withwords.pdf"))
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


quantile(word_position$dist,c(0.05,0.2))
word_new = word_position[word_position$dist<0.41,]
word_new

pdf(paste0(dir,"/Plot4_topic_cluter_centerwords.pdf"))
##plot of topic 
ggg <- ggplot() +
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


words<-data_word
close_word_index <- list()
## cosine similarity
temp <- data.frame(b[,1:2], group)
head(temp)
## center of topic group
temp2 <- data.frame(temp %>% group_by(group) %>% 
                      summarise(x = mean(coordinate_1), y = mean(coordinate_2))) 

head(temp2)




word_position <- data.frame(words,a[,1:2] )
close_word_index <- list()
head(word_position)

library(ggrepel)
pdf(paste0(dir,"/Plot3_topic_close_words.pdf"))

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

##### Cluster_Group near words 

euc.dist <- function(x1, x2) sqrt(sum((x1 - x2) ^ 2))

head(word_position)
pdf(paste0(dir,"/Plot2_cluster_close_words.pdf"))
for(i in 1:nrow(temp2)){
  #i = 1
  ## here 10
  cluster_word_dist<-c()
  close_word_index <-c()
  for(k in c(1:nrow(output_new$z_estimate))){
    cluster_word_dist[k]=euc.dist(temp2[i,-1],output_new$z_estimate[k,])
  }
  close_word_index<-order(as.matrix(cluster_word_dist))[1:25]
  ggg <- ggplot() +
    geom_text(data = b, aes(x=coordinate_1, y = coordinate_2, label = seq(1:ncol(data_set2))),col=as.factor(group), cex=3) +
    geom_point(data = word_position[close_word_index,], aes(x=coordinate_1, y = coordinate_2), cex=1) +
    geom_text(data = temp2[,-1], aes(x=x, y = y, label = paste("Group", seq(1,nrow(temp2)), sep = "")),col="red", cex=3) +
    # geom_point(data = b[i,], aes(x=coordinate_1,y= coordinate_2), col='red', cex=7, shape=2, stroke = 1) +
    scale_color_manual(values = c("black", brewer.pal(n = 9, name = 'Set1'))) +
    labs(title=paste("Cluster", i),
         x ="", y = "")+
    theme_bw() + theme(legend.position = "None") 
  print(ggg)
  ggg1 <- ggg+ geom_label_repel(data = word_position[close_word_index,], aes(x=coordinate_1, y = coordinate_2, label = words),
                                # fontface = 'bold', color = 'white',
                                box.padding = unit(0.5, "lines"),
                                point.padding = unit(0.5, "lines"),
                                segment.color = 'grey50') 
  print(ggg1)
}

dev.off()

save.image("75_result.RData")

