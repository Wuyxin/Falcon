#!/usr/bin/env bash
for DATASET in arxiv cora pubmed citeseer  ; do
    python -m run.eval_nas --log run/nas/$DATASET/log
done



EXP_PATH=run/nas/tu 
for TASK in TU_BZR_MD TU_COX2_MD TU_DHFR_MD TU_ER_MD TU_PTC_MM TU_PTC_MR TU_AIDS TU_Mutagenicity TU_NCI1 TU_NCI109 TU_Tox21_AhR TU_MCF-7 TU_MOLT-4 TU_UACC257 TU_Yeast TU_NCI-H23 TU_OVCAR-8 TU_P388 TU_PC-3 TU_SF-295 TU_SN12C TU_SW-620 ; do 
    python -m run.eval_nas --log $EXP_PATH/$TASK/log
done

