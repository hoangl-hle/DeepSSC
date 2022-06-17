#library
library(ComplexHeatmap) 
library(dplyr)

#-------------------------------o0o-------------------------------#
#                      Extract 50 biomarker genes
#-------------------------------o0o-------------------------------#

top = read.table('matrix_biomarkers.csv', header = T, row.names = NULL, check.names = F, sep = ",")
pam50 = read.table("pam50.csv", check.names = F, header = T, row.names = NULL, sep=",")

top10 = top
colnames(top10) = NULL
top10 = unlist(top10)

#remove duplicated gene
top10[duplicated(top10)]
#[1] "C10orf26" "C10orf26" "C9orf100" "WDR67" 

top10 = unique(top10)
length(top10) #46 genes

#number of genes included in the PAM50 subtype
length(intersect(top10, pam50$pam50)) #3
intersect(top10, pam50$pam50)
#[1] "MIA" "ERBB2" "GRB7" 

#-------------------------------o0o-------------------------------#
#                             Heatmap
#-------------------------------o0o-------------------------------#
exp = read.table("DATA_PAM50_GE_INTER_GE_CNA.csv", check.names = F, header = T, row.names = 1, sep=",")
cli = read.table("clinical_BRCA_TCGA_UCSC_RM_INVALID_CASE.csv", check.names = F, header = T, row.names =1, fill = T,
                 sep=",")
label=read.table("DATA_LABEL_PAM50.csv", check.names = F, header = T, row.names =1, fill = T,
                 sep=",")

exp = exp[rownames(exp) %in% top10,]

cli=cli[rownames(cli) %in% colnames(exp),]

label$patient = rownames(label)
label = label[rownames(label) %in% rownames(cli),]
rownames(label) = label$patient
label = label %>% dplyr::select(-patient)

#heatmap
info2 =as.data.frame(label)
info2 = info2 %>% dplyr::arrange(.$PAM50) #order following groups column
info2$groups = as.character(info2$PAM50)
exp_hm = exp[,rownames(info2)] #change the order of column/patients of exp data following the variable 'info2'
all(colnames(exp_hm) == rownames(info2)) 
#[1] TRUE
exp_hm = as.matrix(exp_hm)

#re-order PAM50 genes
intersect(top10, pam50$pam50)
#"ORC6L" "ERBB2" "GRB7" 
exp_hm = exp_hm[c(4,11,13,1:3,5:10,12,14:nrow(exp_hm)), ]

## Heatmap annotation
library(circlize)
ha = HeatmapAnnotation(subtypes = info2$PAM50,
                       col = list(subtypes = c("Basal" = "turquoise", "Her2" = "pink", "LumA" = "black",
                                               "LumB" = "green", "Normal" = "purple")))

#visualization                
set.seed(53)

hm = Heatmap(exp_hm, name = "expression levels", 
             show_row_names = TRUE, 
             show_column_names = FALSE,
             cluster_rows = FALSE, 
             cluster_columns = FALSE,
             show_row_dend = FALSE,
             show_column_dend=FALSE,
             row_dend_reorder = FALSE, 
             column_dend_reorder = FALSE,
             row_title = "46 biomarker genes", 
             row_title_side = "left", 
             row_title_gp = gpar(fontsize = 11),
             row_names_gp = gpar(col = c(rep("red", 3), rep("black", 46)), fontsize = 8.3),
             column_order = colnames(exp_hm),
             circlize::colorRamp2(c(min(exp_hm), median(exp_hm), max(exp_hm)), c("blue", "white", "red")))
h_list = ha %v% hm
draw(h_list, column_title = "819 patients",column_title_side = "bottom",
     column_title_gp = gpar(fontsize = 11))

#-------------------------------o0o-------------------------------#
#                       Gene Enrichment analysis
#-------------------------------o0o-------------------------------#
library("GOplot")
library("DOSE")
library("clusterProfiler")
library("ggplot2")

#RUN
gene_GO<-top102
eg = bitr(gene_GO, fromType="SYMBOL", toType="ENTREZID",OrgDb="org.Hs.eg.db", drop = TRUE)

target_gene_id = as.character(eg[,2])
display_number = c(1, 13, 21)
#MF
ego_MF <- enrichGO(gene = target_gene_id,
                   OrgDb = org.Hs.eg.db,
                   ont = "MF",
                   pAdjustMethod = "BH",
                   pvalueCutoff = 0.05,
                   qvalueCutoff = 0.05,
                   readable = TRUE)
ego_result_MF <- as.data.frame(ego_MF)[1:display_number[1], ]
write.table(ego_result_MF,'MF.txt', sep = '\t', quote = F, row.names = T, col.names = T)

#CC
ego_CC <- enrichGO(OrgDb="org.Hs.eg.db",
                   gene = target_gene_id,
                   pvalueCutoff = 0.05,
                   ont = "CC",
                   readable=TRUE)
ego_result_CC <- as.data.frame(ego_CC)[1:display_number[2], ]
#>>> No significant terms 

#BP
ego_BP <- enrichGO(OrgDb="org.Hs.eg.db",
                   gene = target_gene_id,
                   pvalueCutoff = 0.05,
                   ont = "BP",
                   readable=TRUE)
ego_result_BP <- na.omit(as.data.frame(ego_BP)[1:display_number[3], ])
write.table(ego_result_BP,'BP.txt', sep = '\t', quote = F, row.names = T, col.names = T)

#KEGG Enrichment
kk<- enrichKEGG(gene= target_gene_id,
                organism = "hsa",keyType = "kegg", pvalueCutoff = 0.15,
                pAdjustMethod = "BH",qvalueCutoff = 0.20)

display_number2 = c(8)
ego_result_kegg <- na.omit(as.data.frame(kk )[1:display_number2[1], ])
write.table(ego_result_kegg,'KEGG.txt', sep = '\t', quote = F, row.names = T, col.names = T)
