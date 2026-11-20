# Usage from the extracted supplement directory
# Rscript code/refit.R retrospective main GAM B M3 1 fitted_values.csv
# Rscript code/refit.R aligned scale2km LM LM Mnet 1 fitted_values.csv
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7L) stop("Expected design, group, family, basis, model, fold and output path")
design <- args[1]; group_name <- args[2]; engine <- args[3]; basis <- args[4]
model_name <- args[5]; fold_n <- args[6]; path <- args[7]
stopifnot(design %in% c("retrospective", "aligned"),
          group_name %in% c("main","scale2km","block4km","block4km_scale2km"),
          model_name %in% c("M0","Mnet","M3","M5","E"), fold_n %in% c(as.character(1:5),"full"))
stopifnot((engine == "GAM" && basis %in% c("A", "B")) || (engine == "LM" && basis == "LM"))
if (design == "aligned") stopifnot(group_name %in% c("main","scale2km"),
    model_name %in% c("Mnet","M3","M5"), basis %in% c("LM","B"), fold_n != "full")
if (model_name == "E") stopifnot(engine == "GAM", group_name == "main", design == "retrospective")
if (fold_n=="full") stopifnot(engine=="GAM", group_name=="main",model_name %in% c("M3","E"))
suppressPackageStartupMessages(library(mgcv))
script <- sub("^--file=", "", grep("^--file=",commandArgs(),value=TRUE)[1])
base <- normalizePath(file.path(dirname(script),".."))
units <- read.csv(file.path(base,"data","units.csv"))
roles <- read.csv(file.path(base,"data","fold_roles.csv"))
if (design=="aligned") {
  replace <- read.csv(file.path(base,"data","aligned_covariates.csv"))
  idx <- match(units$unit_id,replace$unit_id)
  stopifnot(!anyNA(idx),!anyDuplicated(replace$unit_id))
  vars <- setdiff(names(replace),c("grid_m","unit_id"))
  units[vars] <- replace[idx, vars]
}
shares <- paste0("s",c(1,3:8),"_initial")
background <- c("Y0","tmean_initial_c","tmean_change_c","precip_initial_mm","precip_change_mm","elev_mean_m","slope_mean_deg")
transitions <- switch(model_name, M0=character(0), Mnet="forest_net",
                      M3=c("P3_gain","P3_loss"), E=c("P3_gain","P3_loss"), M5=c("P5_gain","P5_loss"))
rhs <- c(shares,background,transitions)
if (engine == "GAM") {
  kb <- if (basis=="B") 10L else 5L
  kt <- if (basis=="B") 8L else 5L
  smooth <- function(x,k) if(length(x)) sprintf("s(%s,k=%d,bs='tp')",x,k) else character(0)
  rhs <- c(shares,smooth(background,kb),smooth(transitions,kt))
  if (model_name=="E") rhs <- c(rhs,sprintf("ti(P3_gain,elev_mean_m,k=c(%d,%d),bs=c('tp','tp'))",kt,kb))
}
formula <- reformulate(rhs, response="Y1")
sel <- roles[roles$group == group_name,]
if (fold_n=="full") {
  train <- units[units$unit_id %in% sel$unit_id,]
  hold <- train
} else {
  sel <- sel[sel$fold==as.integer(fold_n),]
  train <- units[match(sel$unit_id[sel$role=="train"],units$unit_id),]
  hold <- units[match(sel$unit_id[sel$role=="validation"],units$unit_id),]
}
stopifnot(nrow(train)>0,nrow(hold)>0,!anyNA(train),!anyNA(hold))
w <- train$A_g/mean(train$A_g)
fit <- if (engine=="LM") lm(formula,data=train,weights=w) else
  gam(formula,data=train,weights=w,family=gaussian(link="identity"),method="REML",select=FALSE)
if (fold_n=="full") {
  output <- cbind(hold["unit_id"],as.data.frame(predict(fit,newdata=hold,type="terms")))
} else {
  output <- data.frame(unit_id=hold$unit_id,pred=as.numeric(predict(fit,newdata=hold)))
}
write.csv(output,path,row.names=FALSE)
