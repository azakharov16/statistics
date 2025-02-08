Posterior_f <- function(p){
  answer<-0
  if ((p<1)&(p>0))
    answer <- p^3*(1-p)*(0.5+p)
  return (log(answer))}
I <- integrate(Posterior_f, 0, 1)
str(I)
I$Value
g <-function(a,b){
  ans<-a+b
  renurn(ans)
}
const<-I$value
integrate(Posterior_f, 0.5, 1)$value/const
t <- seq(0.1, by=0.01)
qplot(t, Posterior_f(t), geom="line")
x <- rnorm(10^4, mean=1, sd=3)
mean(x)
sum(x>2)/length
res <- Metro_Hastings(Posterior_f, pars=0.5)
Posterior_f(0.5)
res<-Metro_Hastings(Posterior_f,pars=0.5)
str(res)
qplot(res$trace)
p<-res$trace
mean(p)
p
bad<-data.frame(x=c(1,2,3),y=c(0,0,1))
bad
model_logit<-glm(data=bad,y~x,family=binomial(link="logit"))
summary(model_logit)
model_logit2<-MCMClogit(data=bad,y~x,b0=0,B0=0.01)
summary(model_logit2)
str(model_logit2)
qplot(as.vector(model_logit2[,2]))
b2<-as.vector(model_logit2[,2])
mean(b2>0)
mean(b2)
HPDinterval(model_logit2[,2])
HPDinterval(model_logit2[,2],prob=0.9)
library("spikeslab")
h<-cars
qplot(data=h,speed,dist)
h$junk<-rnorm(50)
model<-spikeslab(data=h,dist~speed+junk,n.inter1=1000,n.inter2=4000)
print(model)
library("reshape2")
regressors<-melt(model$model)
head(regressors)
sum(regressors$value==2)/4000
sum(regressors$value==1)/4000
