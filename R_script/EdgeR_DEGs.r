setwd(paste(getwd(),"R_script",sep="/"))
library(reshape2)
library(stringr)
library(ggplot2)
library(edgeR)

times <- read.csv("_times.csv", sep = ",",row.names = 1)
times <- as.numeric(times$Times)
print(paste("Repeats:",times,"times",sep=" "))

count <- read.csv("_count_data.csv", sep = ",",row.names = 1)
clinic_batch <- read.csv("_clinic_batch.csv")
rownames(clinic_batch) <- clinic_batch$Sample.ID
count <- count[,clinic_batch$Sample.ID]
train_samples<- read.csv("_training_samples_inRepeats.csv",row.names = 1)

num = 1
for (a in seq(700,700+3*times-1,3)) {
  print(paste("DEGs by EdgeR: ", num," times"))
  num = num + 1
  a <- paste("X",a,sep="")
  temp_count <- count[, train_samples[,a]]
  temp_clinic <- clinic_batch[train_samples[,a],]
  col <- temp_clinic$Response
  batch <- temp_clinic$batch
  y <- DGEList(counts = temp_count, group = col)
  # Remove low expression
  isexpr <- filterByExpr(y, group=col)
  table(isexpr)
  hasannot <- rowSums(is.na(y$counts))==0
  y <- y[isexpr & hasannot, , keep.lib.sizes=FALSE]
  dge <- calcNormFactors(y)
  if (length(unique(batch)) == 1) {
    design <- model.matrix(~col)
  } else {design <- model.matrix(~batch+col)}
  rownames(design) <- colnames(dge)
  dge <- estimateDisp(dge, design, robust=TRUE)
  fit <- glmQLFit(dge, design, robust=TRUE)
  qlf <- glmQLFTest(fit)
  DEG = topTags(qlf, ,n = nrow(count))
  DEG = as.data.frame(DEG)
  write.csv(DEG,paste("DEGs in repeats/",a,"_edgeR_DEGs.csv",sep=""))
}


