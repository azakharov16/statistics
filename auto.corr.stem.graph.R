auto.corr.stem.graph <- function(x, confidence.inter, max.lag, min.lag = 0){
  library('ggplot2')
  lags <- min.lag:max.lag
  
  x.framed <- data.frame(lag=lags, autocorr=x)
  
  plot <- ggplot(x.framed, aes(x=lag, y=autocorr))
  plot <- plot + ggtitle('Correlogram') + ylab("Autocorrelations")
  plot <- plot + geom_point(stat = "identity")
  
  for(i in 1:length(x)){
    lines <- data.frame(xs=c(i,i),ys=c(0,x[i]))
    plot <- plot + geom_path(data = lines,  aes(x=xs, y=ys))
  }

  plot <- plot + geom_hline(yintercept=-confidence.inter, color = "blue")
  plot <- plot + geom_hline(yintercept=confidence.inter, color = "blue")
  plot <- plot + geom_hline(yintercept = 0, color = "red", size = 0.3)
  plot <- plot + scale_x_continuous(breaks = lags)
  print(plot)
  
}