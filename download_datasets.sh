#!/bin/bash

# 定义数据集列表
DATASETS=(
    "mteb/arguana"
    "mteb/climate-fever"
    "mteb/cqadupstack-retrieval"
    "mteb/dbpedia"
    "mteb/fever"
    "mteb/fiqa"
    "mteb/hotpotqa"
    "mteb/msmarco"
    "mteb/nfcorpus"
    "mteb/scifact"
    "mteb/quora"
    "mteb/scidocs"
    "mteb/touche2020"
    "mteb/trec-covid"
    "deepmind/narrativeqa"
    "dwzhu/LongEmbed"
    "C-MTEB/CmedqaRetrieval"
    "C-MTEB/CovidRetrieval"
    "C-MTEB/DuRetrieval"
    "C-MTEB/EcomRetrieval"
    "C-MTEB/MedicalRetrieval"
    "C-MTEB/MMarcoRetrieval"
    "C-MTEB/T2Retrieval"
    "C-MTEB/VideoRetrieval"
    "maidalun1020/CrosslingualRetrievalQasEn2Zh"
    "maidalun1020/CrosslingualRetrievalLawEn2Zh"
    "maidalun1020/CrosslingualRetrievalBooksEn2Zh"
    "maidalun1020/CrosslingualRetrievalFinanceEn2Zh"
    "maidalun1020/CrosslingualRetrievalPaperEn2Zh"
    "maidalun1020/CrosslingualRetrievalWikiEn2Zh"
    "maidalun1020/CrosslingualRetrievalLawZh2En"
    "maidalun1020/CrosslingualRetrievalBooksZh2En"
    "maidalun1020/CrosslingualRetrievalFinanceZh2En"
    "maidalun1020/CrosslingualRetrievalPaperZh2En"
    "maidalun1020/CrosslingualRetrievalWikiZh2En"
)

# 定义并行作业数量，可以根据你的系统配置调整
PARALLEL_JOBS=8

# 创建一个临时文件，用于存储失败下载的数据集名称
TEMP_FAILED_FILE=$(mktemp)

# 定义下载函数
download_dataset() {
    local dataset=$1
    if ! huggingface-cli download --repo-type dataset "$dataset"; then
        echo "$dataset" >> "$TEMP_FAILED_FILE"
    fi
}

# 导出函数以便子shell可以调用
export -f download_dataset

# 并行执行下载任务
running_jobs=0
for dataset in "${DATASETS[@]}"; do
    # 启动下载任务
    download_dataset "$dataset" &
    running_jobs=$((running_jobs + 1))

    # 如果达到并行作业数量限制，则等待一个任务完成
    if [[ $running_jobs -ge $PARALLEL_JOBS ]]; then
        wait -n
        running_jobs=$((running_jobs - 1))
    fi
done

# 等待所有剩余的后台任务完成
wait

# 检查是否有下载失败的情况
if [ -s "$TEMP_FAILED_FILE" ]; then
    echo "以下数据集下载失败："
    cat "$TEMP_FAILED_FILE"
else
    echo "所有数据集下载成功！"
fi

# 清理临时文件
rm -f "$TEMP_FAILED_FILE"
