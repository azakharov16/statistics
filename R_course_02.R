library("MHadaptive")
library("MCMCpack")
library("ggplot2")
library("spikeslab")
library("reshape2")

# posterior density function
posterior_f <- function(p) {
  answer <- 0
  if ( (p < 1) & (p>0) ) {
            answer <- p^3*(1-p)*(0.5+p)
  }
  # ...
  return(answer)
}

# just test it
posterior_f(-56)
posterior_f(0.5)
posterior_f(0.5,0.6) # two argument is not ok
posterior_f(c(0.5,0.6)) # one vector argument is ok


I <- integrate(posterior_f, 0, 1)
# result of integration is a big list :)
str(I)
I$value # we need only value

t <- seq(0,1, by=0.05)
qplot(t, posterior_f(t), geom="line")

x <- rnorm(10^4, mean=1, sd=3)
mean(x)
sum(x>2)/length(x)

# log of posterior density
log_f <- function(t) {
  retrun(log(posterior_f(t)))
}


res <- Metro_Hastings(log_f, pars=0.5)
str(res)

qplot(res$trace)
p <- res$trace

mean(p)
mean(p>0.5)

head(p)
help(Metro_Hastings)

# bad data for logit model
bad <- data.frame(x=c(1,2,3), y=c(0,0,1))
bad

model_logit <- glm(data=bad, y~x, 
                   family=binomial(link="logit")) 
summary(model_logit)

# here prior is beta ~ N(0, 10^2)
model_logit2 <- MCMClogit(data=bad, y~x,
                          b0=0, B0=0.01)
help(MCMClogit)
summary(model_logit2)

str(model_logit2)
qplot(as.vector(model_logit2[,2]))

b2 <- as.vector(model_logit2[,2])
mean(b2>0)
mean(b2)
HPDinterval(model_logit2[,2],prob = 0.9)

# here prior is beta ~ N(0, 100^2)
model_logit3 <- MCMClogit(data=bad, y~x,
                          b0=0, B0=0.0001)
summary(model_logit3)
HPDinterval(model_logit3[,2],prob = 0.9)


# spike and slab regression, brilliant idea and ugly package
h <- cars
glimpse(h)
qplot(data=h, speed, dist)

h$junk <- rnorm(50)

model <- spikeslab(data=h, dist~speed+junk, 
                   n.iter1 = 1000,
                   n.iter2 = 4000)
print(model)


regressors <- melt(model$model)
head(regressors)
regressors

sum(regressors$value==2)/4000
sum(regressors$value==1)/4000
cor(h$junk, h$dist)
