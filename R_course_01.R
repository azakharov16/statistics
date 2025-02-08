# data file is available at:
# goo.gl/xwOHQW
# installation instructions
# https://github.com/bdemeshev/em301/wiki/R_install

x <- 9
y <- 7

# привет, Маша! # if you can't see cyrillic letters: File-Reopen with encoding - utf8
library("memisc")
library("ggplot2")
library("dplyr") 
library("psych")
library("fortunes")
library("reshape2")
# dplyr и memisc оба определяют функцию select
# нам нужна из пакеты dplyr
# либо dplyr надо загружать после memisc
# либо явно указывать dplyr::select(...)

fortune() # самая важная команда

setwd("~/Downloads") # use Session-Set Working Directory - Choose directory
h <- read.table("flats_moscow.txt", 
                header=TRUE,
                sep="\t",
                dec=".")
# look at the data
glimpse(h)
str(h)
head(h)
tail(h)
h[2,7]
h[2,]
h$price
h$price[3]

mean(h$price)

describe(h)

help("describe")

cos(sin(0)) # Euler tradition 
0 %>% sin() %>% cos() # Magritt tradition

h2 <- filter(.data=h, price>80, brick==1)
help(filter)
glimpse(h2)
glimpse(h)

# если select не работает, то используйте dplyr::select
# дело в том, что select очень частое слово, эту функцию многие пакеты переопределяют
h3 <- select(.data=h2, price, ends_with("sp") )
glimpse(h3)

h4 <- mutate(.data=h3, lprice=log(price))
glimpse(h4)

h4b <- filter(h, price>75, brick==1) %>%
  select(price, ends_with("sp")) %>%
  mutate(lprice=log(price))

glimpse(h)

h5 <- h %>% group_by(code) %>% 
  summarise(av_price=mean(price),
            max_price=max(price),
            min_price=min(price)) 
h5$av_price

h6 <- h %>% group_by(code) %>% 
  mutate(deviation = price - mean(price))

h6

cor(h$totsp, h$kitsp)
model_1 <- lm(data=h, price~totsp+livesp+kitsp)
model_2 <- lm(data=h, price~0+totsp+livesp+kitsp)
report <- summary(model_1)

my_list <- list(a=5, b=10:40, t=h)
str(my_list)
my_list$b

report$residuals
report$coefficients

comparison <- mtable(model_1, model_2)
                     
write.mtable(comparison,forLaTeX=TRUE)

qplot(data=h, x=totsp, 
      y=price, col=kitsp, alpha=0.1) + 
  stat_smooth(method="lm") +
  stat_density2d(col="red") + 
  facet_grid(brick~walk) +
  labs(x="еее")

theme_set(theme_bw())


t <- seq(-10,10, by=0.05)
qplot(t, cos(t), geom="line")

df <- data.frame(t=t)
glimpse(df)
df <- df %>% mutate(cos=cos(t), 
                    sin=sin(t))
glimpse(df)
df_melted <- melt(data=df, id.vars = "t")
glimpse(df_melted)

qplot(data=df_melted, x=t, 
      y=value, col=variable, geom="line")

colnames(df_melted)
colnames(df_melted)[2] <- "function"
