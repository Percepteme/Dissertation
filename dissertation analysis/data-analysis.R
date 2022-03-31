

### Data analysis from inference-analysis-conversion.py pipeline
library(ggplot2)
library(dplyr)

setwd("/home/caiusgibeily/Downloads/Training-Optocamp/test/")

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

  
  ggplot(summary,aes(x=time,y=mean,colour=as.factor(),group=as.factor(pw),fill=as.factor(pw))) +
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
  
