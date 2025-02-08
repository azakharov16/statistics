library("data.table")
library("microbenchmark")
library("lubridate")
library("dplyr")

setwd("~/Downloads/")
players <- read.csv("boxscore.csv")
players <- fread("boxscore.csv")
teams <- fread("team_game_count.csv")

glimpse(players)
glimpse(teams)
players2 <- dplyr::select(players, date, team,
                          minutes, points,
                          fga, fgm)


players3 <- mutate(players2, date = ymd(date))
glimpse(players3)

now() + days(20) + hours(56)
now() - days(1)
weekdays(now())

weekdays(players3$date)

microbenchmark(5/2, 5*0.5)

glimpse(players3)

report <- group_by(players3, date, team) %>%
  summarise(points=sum(points), 
            fga=sum(fga),
            fgm=sum(fgm))
report

system.time(setkey(players3, date, team))

small <- data.frame(x=c(1,2,3,4), 
                    y=c(5,6,7,8))

small
small[3,2]

small2 <- data.table(small)
small2[3,2]
small2
small[,x+y]
small2[,x+y]
small2[3,2*x+y]

small2$num <- 1:nrow(small2)
small3 <- mutate(small, 
                 numrow = 1:nrow(small),
                 pair = numrow %% 2)
small3

str(players3)
report2 <- players3[,list(points=sum(points),
                          fga=sum(fga),
                          fgm=sum(fgm)),
                     by=c("team","date")]
report2
filter(report, team=="DAL", 
       date==ymd("2004-11-02"))
microbenchmark(report2 <- players3[,list(points=sum(points),
                                         fga=sum(fga),
                                         fgm=sum(fgm)),
                                    by=c("team","date")],
               report <- group_by(players3, date, team) %>%
                 summarise(points=sum(points), 
                           fga=sum(fga),
                           fgm=sum(fgm)))

glimpse(players3)

glimpse(players)
pl <- dplyr::select(players, date, team, 
                    player, fgm, fga)
glimpse(pl)
result <- pl %>% group_by(player) %>%
  summarise(prob = sum(fgm)/sum(fga)) 
result
res_sorted <- dplyr::arrange(result, desc(prob) ) 
head(res_sorted, 10)
# help(arrange)

players3

glimpse(players3)
glimpse(teams)
glued <- left_join(players3, teams, 
              by = c("team","date" ))
glimpse(players3)
glimpse(teams)
teams2 <- mutate(teams, date=ymd(date))

glued <- left_join(players3, teams2, 
          by = c("team","date" ))

nrow(glued)
glimpse(glued)

str(players3)
setkey(players3, date, team)
str(teams)
setkey(teams2, date, team)
glued_alt <- players3[teams2]
glimpse(glued_alt)

library("broom")
model <- lm(data=players3, points~fga)
summary(model)
predict(model, players3)

str(model)
glance(model)
tidy(model)
players_augmented <- augment(model, players3 )
glimpse(players_augmented)
glance(model)



