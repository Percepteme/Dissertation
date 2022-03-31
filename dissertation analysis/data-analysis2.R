######## Plotting of all quantitative figures and statistical analysis in R ###### 

### Data analysis from inference-analysis-conversion.py pipeline
library(ggplot2)
library(dplyr)

setwd("...")

dat <- read.csv("NumPulses-Width.csv",sep=",",header= T)
dat1 <- subset(dat,fov!=1)

dat5 <- subset(dat,fov==5)
dat0 <- subset(dat,fov==0)

dat5_4 <- subset(dat5,np==4)
dat5_7 <- subset(dat5,np==7)
dat5_10 <- subset(dat5,np==10)

summary <- dat %>%
  group_by(time,np) %>%
  summarise(
    mean = mean(fluor,na.rm=T),
    sd = sd(fluor,na.rm=T),
    n = n(),
    se = sd / sqrt(n)
  )

  summary5 <- dat5 %>%
  group_by(time,pw) %>%
  summarise(
    mean = mean(fluor,na.rm=T),
    sd = sd(fluor,na.rm=T),
    n = n(),
    se = sd / sqrt(n)
  )

mod <- lm(fluor ~ pw*np,dat)
summary(mod)


  ggplot(summary,aes(x=time,y=mean,colour=as.factor(np),group=as.factor(np),fill=as.factor(np))) +
  geom_point() +
  geom_line(size=1.5) +
  geom_ribbon(aes(x=time,y=mean,ymin=mean-se,ymax=mean+se),alpha=0.4) +
  labs(x = "Time (s)", y = "Normalised fluorescence") +  ## axis labels
  theme(aspect.ratio = 1.5) +   theme_bw() + 
  theme(panel.border = element_blank(),  ## theme commands - use all of these lines to the end
        panel.background = element_blank(),          ## removes background
        panel.grid.major = element_blank(),          ## removes major grid lines
        panel.grid.minor = element_blank(),          ## removes minor grid lines
        axis.line.x = element_line(size = 0.5, linetype = "solid", colour = "black"), ## x axis characteristics
        axis.line.y = element_line(size = 0.5, linetype = "solid", colour = "black"),  ##y axis characteristics
        axis.text.x=element_text(size=10, colour = "black"), ## x axis labels characteristics
        axis.text.y=element_text(size=10, colour = "black"), ## y axis labels characteristics
        axis.title.y=element_text(vjust=0, hjust = 0.5, size=12, margin=margin(0,10,0,0)), ## x axis title characteristics
        axis.title.x=element_text(vjust=0, hjust = 0.5, size=12, margin=margin(10,0,0,0))) +
  guides(color="none") +
  scale_fill_discrete(name ="Number of pulses", limits=c("1","4","7","10"))

  
  ggplot(summary5,aes(x=time,y=mean,group=as.factor(pw),fill=as.factor(pw))) +
    geom_point() +
    geom_line(size=1.5) +
    geom_ribbon(aes(x=time,y=mean,ymin=mean-se,ymax=mean+se),alpha=0.4) +
    labs(x = "Time (s)", y = "Normalised fluorescence") +  ## axis labels
    theme(aspect.ratio = 1.5) +   theme_bw() + 
    theme(panel.border = element_blank(),  ## theme commands - use all of these lines to the end
          panel.background = element_blank(),          ## removes background
          panel.grid.major = element_blank(),          ## removes major grid lines
          panel.grid.minor = element_blank(),          ## removes minor grid lines
          axis.line.x = element_line(size = 0.5, linetype = "solid", colour = "black"), ## x axis characteristics
          axis.line.y = element_line(size = 0.5, linetype = "solid", colour = "black"),  ##y axis characteristics
          axis.text.x=element_text(size=10, colour = "black"), ## x axis labels characteristics
          axis.text.y=element_text(size=10, colour = "black"), ## y axis labels characteristics
          axis.title.y=element_text(vjust=0, hjust = 0.5, size=12, margin=margin(0,10,0,0)), ## x axis title characteristics
          axis.title.x=element_text(vjust=0, hjust = 0.5, size=12, margin=margin(10,0,0,0))) +
    guides(color="none") +
    scale_fill_discrete(name ="Pulse width", limits=c("1960","3960","6960"))
  
  
  #########################
  ## Trial 2
  setwd("...")
  
  dat <- read.csv("session_1-PW-NP-FOVs_0-1-4-5.csv",sep=",",header= T)
  
  summary <- dat %>%
    group_by(np) %>%
    summarise(
      mean = mean(fluor,na.rm=T),
      sd = sd(fluor,na.rm=T),
      n = n(),
      se = sd / sqrt(n)
    )
  
  ggplot(summary,aes(x=time,y=mean,colour=as.factor(pw),group=as.factor(pw),fill=as.factor(pw))) +
    #geom_point(data=dat,aes(x=time,y=fluor,colour=as.factor(np))) +
    geom_line(size=1.5) +
    geom_ribbon(aes(x=time,y=mean,ymin=mean-se,ymax=mean+se),alpha=0.4) +
    labs(x = "Time (s)", y = "Normalised fluorescence") +  ## axis labels
    theme(aspect.ratio = 1.5) +   theme_bw() + 
    theme(panel.border = element_blank(),  ## theme commands - use all of these lines to the end
          panel.background = element_blank(),          ## removes background
          panel.grid.major = element_blank(),          ## removes major grid lines
          panel.grid.minor = element_blank(),          ## removes minor grid lines
          axis.line.x = element_line(size = 1, linetype = "solid", colour = "black"), ## x axis characteristics
          axis.line.y = element_line(size = 1, linetype = "solid", colour = "black"),  ##y axis characteristics
          axis.text.x=element_text(size=15, colour = "black"), ## x axis labels characteristics
          axis.text.y=element_text(size=15, colour = "black"), ## y axis labels characteristics
          axis.title.y=element_text(vjust=0, hjust = 0.5, size=15, margin=margin(0,10,0,0)), ## x axis title characteristics
          axis.title.x=element_text(vjust=0, hjust = 0.5, size=15, margin=margin(10,0,0,0))) +
    guides(color="none")
    #scale_fill_discrete(name ="Number of pulses", limits=c("1","4","7","10"))
  
  
dat$np = as.factor(dat$np)
dat$pw = as.factor(dat$pw)
mod = aov(mean ~ pw + np,summary)

summary(mod)  
TukeyHSD(mod)
###########################
### Peak amp and bd for parameter search
setwd("...")

dat <- read.csv("data-np-pw.csv",sep=",",header= T)
dat$np = as.factor(dat$np)
dat$pw = as.factor(dat$pw)

ggplot(dat,aes(x=np,y=maxv,fill=pw)) +
  geom_point(aes(color=pw),position=position_jitterdodge()) +
  geom_boxplot(alpha=0.6) +
  #geom_text(data=agg, aes(label = slope, y = slope + 0.008,group=pw)) +
  theme(aspect.ratio = 1.5) +   theme_bw() + 
  theme(panel.border = element_blank(),  ## theme commands - use all of these lines to the end
        panel.background = element_blank(),          ## removes background
        panel.grid.major = element_blank(),          ## removes major grid lines
        panel.grid.minor = element_blank(),          ## removes minor grid lines
        axis.line.x = element_line(size = 1, linetype = "solid", colour = "black"), ## x axis characteristics
        axis.line.y = element_line(size = 1, linetype = "solid", colour = "black"),  ##y axis characteristics
        axis.text.x=element_text(size=16, colour = "black",family="arial"), ## x axis labels characteristics
        axis.text.y=element_text(size=16, colour = "black",family="arial"), ## y axis labels characteristics
        axis.title.y=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(0,10,0,0),family="arial"), ## x axis title characteristics
        axis.title.x=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(10,0,0,0),family="arial"),
        legend.text = element_text(size=15),
        legend.title = element_text(size=15)) +
  labs(x = "Number of pulses", y = "Peak amplitude (normalised r.f.)") +
  guides(color="none", fill=guide_legend(title="Pulse width (ms)"))

### burst duration ####


ggplot(dat,aes(x=np,y=dur,fill=pw)) +
  geom_point(aes(color=pw),position=position_jitterdodge()) +
  geom_boxplot(alpha=0.6) +
  #geom_text(data=agg, aes(label = slope, y = slope + 0.008,group=pw)) +
  theme(aspect.ratio = 1.5) +   theme_bw() + 
  theme(panel.border = element_blank(),  ## theme commands - use all of these lines to the end
        panel.background = element_blank(),          ## removes background
        panel.grid.major = element_blank(),          ## removes major grid lines
        panel.grid.minor = element_blank(),          ## removes minor grid lines
        axis.line.x = element_line(size = 1, linetype = "solid", colour = "black"), ## x axis characteristics
        axis.line.y = element_line(size = 1, linetype = "solid", colour = "black"),  ##y axis characteristics
        axis.text.x=element_text(size=16, colour = "black",family="arial"), ## x axis labels characteristics
        axis.text.y=element_text(size=16, colour = "black",family="arial"), ## y axis labels characteristics
        axis.title.y=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(0,10,0,0),family="arial"), ## x axis title characteristics
        axis.title.x=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(10,0,0,0),family="arial"),
        legend.text = element_text(size=15),
        legend.title = element_text(size=15)) +
  labs(x = "Number of pulses", y = "Peak amplitude (normalised r.f.)") +
  guides(color="none", fill=guide_legend(title="Pulse width (ms)"))



summary <- dat %>%
  group_by(pw,np) %>%
  summarise(
    mean = mean(maxv,na.rm=T),
    sd = sd(maxv,na.rm=T),
    n = n(),
    se = sd / sqrt(n)
  )
mod <- aov(maxv ~ pw,dat)
summary(mod)
TukeyHSD(mod)
###############################################################
dat2 = subset(dat,np!="1")

summary <- dat2 %>%
  group_by(pw) %>%
  summarise(
    mean = mean(dur,na.rm=T),
    sd = sd(dur,na.rm=T),
    n = n(),
    se = sd / sqrt(n)
  )
mod <- anova(maxv ~ pw + np,dat2)

mod <- anova(lm(dur ~ pw*np, data = dat2))

summary(mod)

setwd("...")

dat <- read.csv("before-after-average.csv",sep=",",header= T)
dat$drug <- as.factor(dat$drug)
dat$fov <- as.factor(dat$fov)

dat$slope <- as.numeric(dat$slope,na.rm=TRUE)

dat0 <- subset(dat,drug=="0")
dat25 <- subset(dat,drug=="25")
dat4 <- subset(dat,np==4)


slope <- dat %>%
  group_by(fov,drug) %>%
  summarise(
    mean = round(mean(slope,na.rm=T),2),
    sd = sd(slope,na.rm=T),
    n = n(),
    se = sd / sqrt(n)
  )

agg = round(aggregate(slope ~  np + pw, dat, mean),3)

############### Low-level analysis - drug effect ####
setwd("...")

dat <- read.csv("before-after-average.csv",sep=",",header= T)
dat$drug <- as.factor(dat$drug)
dat$fov <- as.factor(dat$fov)

dat$slope <- as.numeric(dat$slope,na.rm=TRUE)

dat0 <- subset(dat,drug=="0")
dat25 <- subset(dat,drug=="25")
### slope ####
ggplot(dat,aes(x=np,y=slope,fill=pw)) +
  geom_point(aes(color=pw),position=position_jitterdodge()) +
  geom_boxplot(alpha=0.6) +
  #geom_text(data=agg, aes(label = slope, y = slope + 0.008,group=pw)) +
  theme(aspect.ratio = 1.5) +   theme_bw() + 
  theme(panel.border = element_blank(),  ## theme commands - use all of these lines to the end
        panel.background = element_blank(),          ## removes background
        panel.grid.major = element_blank(),          ## removes major grid lines
        panel.grid.minor = element_blank(),          ## removes minor grid lines
        axis.line.x = element_line(size = 0.5, linetype = "solid", colour = "black"), ## x axis characteristics
        axis.line.y = element_line(size = 0.5, linetype = "solid", colour = "black"),  ##y axis characteristics
        axis.text.x=element_text(size=14, colour = "black",family="arial"), ## x axis labels characteristics
        axis.text.y=element_text(size=14, colour = "black",family="arial"), ## y axis labels characteristics
        axis.title.y=element_text(vjust=0, hjust = 0.5, size=14, margin=margin(0,10,0,0),family="arial"), ## x axis title characteristics
        axis.title.x=element_text(vjust=0, hjust = 0.5, size=14, margin=margin(10,0,0,0),family="arial"),
        legend.text = element_text(size=14),
        legend.title = element_text(size=14)) +
  labs(x = "Number of pulses at 10Hz", y = "Decay gradient (r.f./s)") +
  guides(color="none", fill=guide_legend(title="Pulse duration (ms)")) +
  ylim(-0.2,0.1)

### amplitude ###

ggplot(dat,aes(x=fov,y=maxv,fill=drug)) +
  geom_point(aes(color=drug),position=position_jitterdodge()) +
  geom_boxplot(alpha=0.6) +
  #geom_text(data=agg, aes(label = slope, y = slope + 0.008,group=pw)) +
  theme(aspect.ratio = 1.5) +   theme_bw() + 
  theme(panel.border = element_blank(),  ## theme commands - use all of these lines to the end
        panel.background = element_blank(),          ## removes background
        panel.grid.major = element_blank(),          ## removes major grid lines
        panel.grid.minor = element_blank(),          ## removes minor grid lines
        axis.line.x = element_line(size = 1, linetype = "solid", colour = "black"), ## x axis characteristics
        axis.line.y = element_line(size = 1, linetype = "solid", colour = "black"),  ##y axis characteristics
        axis.text.x=element_text(size=16, colour = "black",family="arial"), ## x axis labels characteristics
        axis.text.y=element_text(size=16, colour = "black",family="arial"), ## y axis labels characteristics
        axis.title.y=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(0,10,0,0),family="arial"), ## x axis title characteristics
        axis.title.x=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(10,0,0,0),family="arial"),
        legend.text = element_text(size=15),
        legend.title = element_text(size=15)) +
  labs(x = "Field of view (FOV)", y = "Peak amplitude (normalised r.f.)") +
  guides(color="none", fill=guide_legend(title="Concentration (μM)")) +
  ylim(0,0.4)
  
### burst duration ####


ggplot(dat,aes(x=fov,y=dur,fill=drug)) +
  geom_point(aes(color=drug),position=position_jitterdodge()) +
  geom_boxplot(alpha=0.6) +
  #geom_text(data=agg, aes(label = slope, y = slope + 0.008,group=pw)) +
  theme(aspect.ratio = 1.5) +   theme_bw() + 
  theme(panel.border = element_blank(),  ## theme commands - use all of these lines to the end
        panel.background = element_blank(),          ## removes background
        panel.grid.major = element_blank(),          ## removes major grid lines
        panel.grid.minor = element_blank(),          ## removes minor grid lines
        axis.line.x = element_line(size = 1, linetype = "solid", colour = "black"), ## x axis characteristics
        axis.line.y = element_line(size = 1, linetype = "solid", colour = "black"),  ##y axis characteristics
        axis.text.x=element_text(size=16, colour = "black",family="arial"), ## x axis labels characteristics
        axis.text.y=element_text(size=16, colour = "black",family="arial"), ## y axis labels characteristics
        axis.title.y=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(0,10,0,0),family="arial"), ## x axis title characteristics
        axis.title.x=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(10,0,0,0),family="arial"),
        legend.text = element_text(size=15),
        legend.title = element_text(size=15)) +
  labs(x = "Field of view (FOV)", y = "Burst duration (s)") +
  guides(color="none", fill=guide_legend(title="Concentration (μM)")) + 
  ylim(0,4)



########## Aβ treatment #############
setwd("...")

dat <- read.csv("abeta-dur.csv",sep=",",header= T)
dat$conc <- as.factor(dat$conc)

dat0 <- subset(dat,conc=="0")
dat02 <- subset(dat,conc=="0.2")
dat2 <- subset(dat,conc=="2")
ggplot(dat,aes(x=conc,y=dur,fill="coral",color="coral")) +
  geom_jitter(width=0.3) +
  geom_boxplot(alpha=0.6,color="black") +
  #geom_text(data=agg, aes(label = slope, y = slope + 0.008,group=pw)) +
  theme(aspect.ratio = 1.5) +   theme_bw() + 
  theme(panel.border = element_blank(),  ## theme commands - use all of these lines to the end
        panel.background = element_blank(),          ## removes background
        panel.grid.major = element_blank(),          ## removes major grid lines
        panel.grid.minor = element_blank(),          ## removes minor grid lines
        axis.line.x = element_line(size = 1, linetype = "solid", colour = "black"), ## x axis characteristics
        axis.line.y = element_line(size = 1, linetype = "solid", colour = "black"),  ##y axis characteristics
        axis.text.x=element_text(size=16, colour = "black",family="arial"), ## x axis labels characteristics
        axis.text.y=element_text(size=16, colour = "black",family="arial"), ## y axis labels characteristics
        axis.title.y=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(0,10,0,0),family="arial"), ## x axis title characteristics
        axis.title.x=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(10,0,0,0),family="arial"),
        legend.text = element_text(size=15),
        legend.title = element_text(size=15)) +
  guides(color="none",fill="none") +
  labs(x = "Concentration (nM)", y = "Burst duration (s)") +
  ylim(c(0,3))
t.test(dat0$dur,dat2$dur)

ggplot(dat,aes(x=treat,y=diff,fill="coral",color="coral")) +
  geom_jitter(width=0.3) +
  geom_boxplot(alpha=0.6,color="black") +
  #geom_text(data=agg, aes(label = slope, y = slope + 0.008,group=pw)) +
  theme(aspect.ratio = 1.5) +   theme_bw() + 
  theme(panel.border = element_blank(),  ## theme commands - use all of these lines to the end
        panel.background = element_blank(),          ## removes background
        panel.grid.major = element_blank(),          ## removes major grid lines
        panel.grid.minor = element_blank(),          ## removes minor grid lines
        axis.line.x = element_line(size = 1, linetype = "solid", colour = "black"), ## x axis characteristics
        axis.line.y = element_line(size = 1, linetype = "solid", colour = "black"),  ##y axis characteristics
        axis.text.x=element_text(size=16, colour = "black",family="arial"), ## x axis labels characteristics
        axis.text.y=element_text(size=16, colour = "black",family="arial"), ## y axis labels characteristics
        axis.title.y=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(0,10,0,0),family="arial"), ## x axis title characteristics
        axis.title.x=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(10,0,0,0),family="arial"),
        legend.text = element_text(size=15),
        legend.title = element_text(size=15)) +
  guides(color="none",fill="none") +
  labs(x = "Concentration (nM)", y = "Peak amplitude (normalised r.f.)")

#######
## Midlevel correlation analysis
setwd("...")

dat <- read.csv("correlation_averages.csv",sep=",",header= T)
dat$conc <- as.factor(dat$conc)
dat$fov <- as.factor(dat$fov)
dat$ap <- as.factor(dat$ap)

ggplot(dat,aes(x=conc,y=correlation,fill=ap,color=ap)) +
  geom_point(position=position_jitterdodge()) +
  geom_boxplot(alpha=0.6,color="black") +
  #geom_text(data=agg, aes(label = slope, y = slope + 0.008,group=pw)) +
  theme(aspect.ratio = 1.5) +   theme_bw() + 
  theme(panel.border = element_blank(),  ## theme commands - use all of these lines to the end
        panel.background = element_blank(),          ## removes background
        panel.grid.major = element_blank(),          ## removes major grid lines
        panel.grid.minor = element_blank(),          ## removes minor grid lines
        axis.line.x = element_line(size = 1, linetype = "solid", colour = "black"), ## x axis characteristics
        axis.line.y = element_line(size = 1, linetype = "solid", colour = "black"),  ##y axis characteristics
        axis.text.x=element_text(size=16, colour = "black",family="arial"), ## x axis labels characteristics
        axis.text.y=element_text(size=16, colour = "black",family="arial"), ## y axis labels characteristics
        axis.title.y=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(0,10,0,0),family="arial"), ## x axis title characteristics
        axis.title.x=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(10,0,0,0),family="arial"),
        legend.text = element_text(size=15),
        legend.title = element_text(size=15)) +
  scale_fill_discrete(name="Aperture diameter", limits = c("0","1"),labels=c("Constricted","Open")) + 
  guides(color = "none") +
  labs(x = "Concentration of Aβ42 (nM)", y = "Average pairwise correlation") +
  ylim(c(0,1))

####### CAITCHA ###########
setwd("/home/caiusgibeily/Downloads/Training-Optocamp/test/CAITCHA/")
dat <- read.csv("performance.csv",sep=",",header= T)

m12 = subset(dat,dat$part=="m12")
m18 = subset(dat,dat$part=="m18")
p1 = subset(dat,dat$part=="m1")
p2 = subset(dat,dat$part=="m2")
p3 = subset(dat,dat$part=="p3")
mod = aov(error ~ part, dat)
summary(mod)

summary <- dat %>%
  group_by(part,status) %>%
  summarise(
    mean = mean(error,na.rm=T),
    sd = sd(error,na.rm=T),
    n = n(),
    se = sd / sqrt(n)
  )

ggplot(dat,aes(x=part,y=error,color=status,fill=status)) +
  geom_jitter(width=0.2) +
  geom_bar(stat="summary",fun="mean",alpha=0.4) +
  geom_errorbar(data=summary,aes(x=part,y=mean,ymin=mean-se,ymax=mean+se),width=0.2) +
  #geom_boxplot(alpha=0.6,color="black") +
  #geom_text(data=agg, aes(label = slope, y = slope + 0.008,group=pw)) +
  theme(aspect.ratio = 1.5) +   theme_bw() + 
  theme(panel.border = element_blank(),  ## theme commands - use all of these lines to the end
        panel.background = element_blank(),          ## removes background
        panel.grid.major = element_blank(),          ## removes major grid lines
        panel.grid.minor = element_blank(),          ## removes minor grid lines
        axis.line.x = element_line(size = 1, linetype = "solid", colour = "black"), ## x axis characteristics
        axis.line.y = element_line(size = 1, linetype = "solid", colour = "black"),  ##y axis characteristics
        axis.text.x=element_text(size=16, colour = "black",family="arial"), ## x axis labels characteristics
        axis.text.y=element_text(size=16, colour = "black",family="arial"), ## y axis labels characteristics
        axis.title.y=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(0,10,0,0),family="arial"), ## x axis title characteristics
        axis.title.x=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(10,0,0,0),family="arial"),
        legend.text = element_text(size=15),
        legend.title = element_text(size=15)) +
  guides(color="none") +
  scale_fill_discrete(name="Participant",limits=c("human","machine"),labels=c("Human","Machine")) +
  labs(x = "Participant", y = "Mean absolute error (ME)") +
  ylim(c(0,1))

############################# Num ROIs

setwd("...")
dat <- read.csv("completion_times.csv",sep=",",header= T)

m12 = subset(dat,dat$part=="m12")
m18 = subset(dat,dat$part=="m18")
p1 = subset(dat,dat$part=="m1")
p2 = subset(dat,dat$part=="m2")
p3 = subset(dat,dat$part=="p3")

mod = aov(rois ~ part, dat)
summary(mod)

summary <- dat %>%
  group_by(status,part) %>%
  summarise(
    mean = mean(rois,na.rm=T),
    sd = sd(rois,na.rm=T),
    n = n(),
    se = sd / sqrt(n)
  )

ggplot(dat,aes(x=part,y=rois,color=status,fill=status)) +
  geom_jitter(width=0.2) +
  geom_bar(stat="summary",fun="mean",alpha=0.4) +
  geom_errorbar(data=summary,aes(x=part,y=mean,ymin=mean-se,ymax=mean+se),width=0.2) +
  #geom_boxplot(alpha=0.6,color="black") +
  #geom_text(data=agg, aes(label = slope, y = slope + 0.008,group=pw)) +
  theme(aspect.ratio = 1.5) +   theme_bw() + 
  theme(panel.border = element_blank(),  ## theme commands - use all of these lines to the end
        panel.background = element_blank(),          ## removes background
        panel.grid.major = element_blank(),          ## removes major grid lines
        panel.grid.minor = element_blank(),          ## removes minor grid lines
        axis.line.x = element_line(size = 1, linetype = "solid", colour = "black"), ## x axis characteristics
        axis.line.y = element_line(size = 1, linetype = "solid", colour = "black"),  ##y axis characteristics
        axis.text.x=element_text(size=16, colour = "black",family="arial"), ## x axis labels characteristics
        axis.text.y=element_text(size=16, colour = "black",family="arial"), ## y axis labels characteristics
        axis.title.y=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(0,10,0,0),family="arial"), ## x axis title characteristics
        axis.title.x=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(10,0,0,0),family="arial"),
        legend.text = element_text(size=15),
        legend.title = element_text(size=15)) +
  guides(color="none",fill="none") +
  scale_fill_discrete(name="Participant",limits=c("human","machine"),labels=c("Human","Machine")) +
  labs(x = "Labeller", y = "Number of ROIs counted")

################## Times
dat2 = subset(dat,part!="ref")

summary <- dat2 %>%
  group_by(status) %>%
  summarise(
    mean = mean(time,na.rm=T),
    sd = sd(time,na.rm=T),
    n = n(),
    se = sd / sqrt(n)
  )

ggplot(dat2,aes(x=part,y=time,color=status,fill=status)) +
  geom_jitter(width=0.2) +
  geom_bar(stat="summary",fun="mean",alpha=0.4) +
  geom_errorbar(data=summary,aes(x=part,y=mean,ymin=mean-se,ymax=mean+se),width=0.2) +
  #geom_boxplot(alpha=0.6,color="black") +
  #geom_text(data=agg, aes(label = slope, y = slope + 0.008,group=pw)) +
  theme(aspect.ratio = 1.5) +   theme_bw() + 
  theme(panel.border = element_blank(),  ## theme commands - use all of these lines to the end
        panel.background = element_blank(),          ## removes background
        panel.grid.major = element_blank(),          ## removes major grid lines
        panel.grid.minor = element_blank(),          ## removes minor grid lines
        axis.line.x = element_line(size = 1, linetype = "solid", colour = "black"), ## x axis characteristics
        axis.line.y = element_line(size = 1, linetype = "solid", colour = "black"),  ##y axis characteristics
        axis.text.x=element_text(size=16, colour = "black",family="arial"), ## x axis labels characteristics
        axis.text.y=element_text(size=16, colour = "black",family="arial"), ## y axis labels characteristics
        axis.title.y=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(0,10,0,0),family="arial"), ## x axis title characteristics
        axis.title.x=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(10,0,0,0),family="arial"),
        legend.text = element_text(size=15),
        legend.title = element_text(size=15)) +
  guides(color="none") +
  scale_fill_discrete(name="Participant",limits=c("human","machine"),labels=c("Human","Machine")) +
  labs(x = "Labeller", y = "Mean absolute error (ME)")

############# per ROI

ggplot(dat2,aes(x=img,y=timeroi,color=status,fill=status,group=part)) +
  geom_point(aes(shape=part),size=2) +
  geom_line(size=1) +
  #geom_boxplot(alpha=0.6,color="black") +
  #geom_text(data=agg, aes(label = slope, y = slope + 0.008,group=pw)) +
  theme(aspect.ratio = 1.5) +   theme_bw() + 
  theme(panel.border = element_blank(),  ## theme commands - use all of these lines to the end
        panel.background = element_blank(),          ## removes background
        panel.grid.major = element_blank(),          ## removes major grid lines
        panel.grid.minor = element_blank(),          ## removes minor grid lines
        axis.line.x = element_line(size = 1, linetype = "solid", colour = "black"), ## x axis characteristics
        axis.line.y = element_line(size = 1, linetype = "solid", colour = "black"),  ##y axis characteristics
        axis.text.x=element_text(size=16, colour = "black",family="arial"), ## x axis labels characteristics
        axis.text.y=element_text(size=16, colour = "black",family="arial"), ## y axis labels characteristics
        axis.title.y=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(0,10,0,0),family="arial"), ## x axis title characteristics
        axis.title.x=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(10,0,0,0),family="arial"),
        legend.text = element_text(size=15),
        legend.title = element_text(size=15)) +
  guides(fill="none") +
  scale_color_discrete(name="Participant",limits=c("human","machine"),labels=c("Human","Machine")) +
  scale_shape_discrete(name="Participant ID",limits=c("m12","m18","p1","p2","p3"),
                       labels=c("Machine-12","Machine-18","Participant 1", "Participant 2","Participant 3")) +
  labs(x = "Ordered images", y = "Label time per ROI (s)")

hdat = subset(dat2,status=="human")
adaptation = aov(timeroi ~ img, hdat)
summary(adaptation)
###########
summary <- dat2 %>%
  group_by(status,part) %>%
  summarise(
    mean = mean(roidif,na.rm=T),
    sd = sd(roidif,na.rm=T),
    n = n(),
    se = sd / sqrt(n)
  )


ggplot(dat2,aes(x=part,y=roidif,color=status,fill=status)) +
  geom_jitter(width=0.2) +
  geom_bar(stat="summary",fun="mean",alpha=0.4) +
  geom_errorbar(data=summary,aes(x=part,y=mean,ymin=mean-se,ymax=mean+se),width=0.2) +
  #geom_boxplot(alpha=0.6,color="black") +
  #geom_text(data=agg, aes(label = slope, y = slope + 0.008,group=pw)) +
  theme(aspect.ratio = 1.5) +   theme_bw() + 
  theme(panel.border = element_blank(),  ## theme commands - use all of these lines to the end
        panel.background = element_blank(),          ## removes background
        panel.grid.major = element_blank(),          ## removes major grid lines
        panel.grid.minor = element_blank(),          ## removes minor grid lines
        axis.line.x = element_line(size = 1, linetype = "solid", colour = "black"), ## x axis characteristics
        axis.line.y = element_line(size = 1, linetype = "solid", colour = "black"),  ##y axis characteristics
        axis.text.x=element_text(size=16, colour = "black",angle=90,family="arial"), ## x axis labels characteristics
        axis.text.y=element_text(size=16, colour = "black",family="arial"), ## y axis labels characteristics
        axis.title.y=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(0,10,0,0),family="arial"), ## x axis title characteristics
        axis.title.x=element_text(vjust=0, hjust = 0.5, size=16, margin=margin(10,0,0,0),family="arial"),
        legend.text = element_text(size=15),
        legend.title = element_text(size=15)) +
  guides(color="none") +
  scale_fill_discrete(name="Participant",limits=c("human","machine"),labels=c("Human","Machine")) +
  labs(x = "Labeller", y = "Difference in ROIs counted")

