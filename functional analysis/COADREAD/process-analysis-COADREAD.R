#library
library(ComplexHeatmap)
library(dplyr)

#-------------------------------o0o-------------------------------#
#                      Extract 50 biomarker genes
#-------------------------------o0o-------------------------------#

top = read.table('matrix_biomarkers.csv', header = T, row.names = NULL, check.names = F, sep = ",")

top10 = top[1:10,]
colnames(top10) = NULL
top10 = unlist(top10)

#remove duplicated gene
top10[duplicated(top10)]
#[1] "ODZ3"    "WHAMML1"

top10 = unique(top10)
length(top10) #38 genes

#write.table(top10, 'top10_standardized.txt', sep = '\t', row.names = F, col.names = T)

#-------------------------------o0o-------------------------------#
#                             Heatmap
#-------------------------------o0o-------------------------------#
exp = read.table("df_GE_labeled.csv", check.names = F, header = T, row.names = 1, sep=",") %>% t()
label=read.table("df_labeled.csv", check.names = F, header = T, row.names =1, fill = T,
                 sep=",")

exp = exp[rownames(exp) %in% top10,]

colnames(label)[1] = 'label'
exp_hm = exp
#heatmap
info2 =as.data.frame(label)
info2 = info2 %>% dplyr::arrange(.$label) #order following groups column
info2$groups = as.character(info2$label)
exp_hm = exp[,rownames(info2)] #change the order of column/patients of exp data following the variable 'info2'
all(colnames(exp_hm) == rownames(info2))
#[1] TRUE
exp_hm = as.matrix(exp_hm)

## Heatmap annotation
library(circlize)
ha = HeatmapAnnotation(subtypes = info2$groups,
                       col = list(subtypes = c("CMS1" = "turquoise", "CMS2" = "pink", "CMS3" = "black",
                                               "CMS4" = "green")))

#visualization

hm = Heatmap(exp_hm, name = "expression levels",
             show_row_names = TRUE,
             show_column_names = FALSE,
             cluster_rows = FALSE,
             cluster_columns = FALSE,
             show_row_dend = FALSE,
             show_column_dend=FALSE,
             row_dend_reorder = FALSE,
             column_dend_reorder = FALSE,
             row_title = "38 biomarker genes",
             row_title_side = "left",
             row_title_gp = gpar(fontsize = 11),
             row_names_gp = gpar(fontsize = 8.4),
             column_order = colnames(exp_hm),
             circlize::colorRamp2(c(min(exp_hm), median(exp_hm), max(exp_hm)), c("blue", "white", "red")))
h_list = ha %v% hm
draw(h_list, column_title = "264 patients",column_title_side = "bottom",
     column_title_gp = gpar(fontsize = 11))


#-------------------------------o0o-------------------------------#
#                       Gene Enrichment analysis
#-------------------------------o0o-------------------------------#
library("GOplot")
library("DOSE")
library("clusterProfiler")
library("ggplot2")

gene_GO<-top10
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
write.table(ego_result_CC,'CC.txt', sep = '\t', quote = F, row.names = T, col.names = T)

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

