#!/bin/bash

# 定义日志文件路径
LOG_FILE="test_result.log"
MMLU_DIR="MMLU"

# 清空或创建日志文件
echo "=== 测试开始 ===" > $LOG_FILE
date >> $LOG_FILE

# PART1 显存峰值、推理速度测试
echo "PART1显存峰值、推理速度测试" | tee -a $LOG_FILE
echo "等待10秒后开始测试..." | tee -a $LOG_FILE
sleep 10
python qwen.test.py >> $LOG_FILE 2>&1
echo "PART1测试完成" | tee -a $LOG_FILE

# PART2 知识保留能力测试
echo "PART2知识保留能力测试" | tee -a $LOG_FILE

# 进入MMLU目录
cd $MMLU_DIR || { echo "无法进入MMLU目录"; exit 1; }

# 测试xzr缓存类型
echo "测试xzr缓存类型，等待10秒..." | tee -a ../$LOG_FILE
sleep 10
python evaluate_qwen.py --cache_type xzr --save-dir xzr_results >> ../$LOG_FILE 2>&1

# 测试windows缓存类型
echo "测试windows缓存类型，等待10秒..." | tee -a ../$LOG_FILE
sleep 10
python evaluate_qwen.py --cache_type windows --save-dir windows_results >> ../$LOG_FILE 2>&1

# 测试full缓存类型
echo "测试full缓存类型，等待10秒..." | tee -a ../$LOG_FILE
sleep 10
python evaluate_qwen.py --cache_type full --save-dir full_results >> ../$LOG_FILE 2>&1

# 返回上级目录
cd ..

echo "=== 所有测试完成 ===" | tee -a $LOG_FILE
date >> $LOG_FILE
