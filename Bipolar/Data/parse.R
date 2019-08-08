# Data Source:  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81904
# More Detail (starting on page 21): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5003425/pdf/nihms807574.pdf

load("GSE81904_bipolar_data_Cell2016.Rdata")

# bipolar_dge is the raw data (44994 observations with 24904 features)
# pca.load is are the principle components of the 13166 features that passed the selection process
# pca.scores is the representation of the 27499 observations that passed the selection process 
#  -  This matches the data used by scvis for their experiments

write.table(pca.load, file = "bipolar_pc.tsv", sep="\t",  col.names = FALSE, row.names = FALSE)
write.table(pca.scores, file = "bipolar_rep.tsv", sep="\t",  col.names = FALSE, row.names = FALSE)
