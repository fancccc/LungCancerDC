#!/bin/bash

# 初始化GPU状态标志
gpu0_busy=true
gpu1_busy=true

# 循环检查GPU状态，直到找到空闲的GPU
while $gpu0_busy || $gpu1_busy; do
    # 检查GPU 0
    if ! nvidia-smi -i 0 | grep 'No running processes found' > /dev/null; then
        echo "GPU 0 is busy."
        gpu0_busy=true
    else
        echo "GPU 0 is free."
        gpu0_busy=false
    fi

    # 检查GPU 1
    if ! nvidia-smi -i 1 | grep 'No running processes found' > /dev/null; then
        echo "GPU 1 is busy."
        gpu1_busy=true
    else
        echo "GPU 1 is free."
        gpu1_busy=false
    fi

    # 如果两个GPU都忙，等待一段时间后再次检查
    if $gpu0_busy && $gpu1_busy; then
        echo "All GPUs are busy. Waiting..."
        sleep 60  # 等待60秒
    else
        # 如果找到空闲的GPU，跳出循环
        break
    fi
done

# 使用空闲的GPU执行程序
if ! $gpu0_busy; then
    echo "Using GPU 0."
    CUDA_VISIBLE_DEVICES=0 python train_base.py --epochs 300 --batch-size 1 --rd 34 --lr 0.0001
elif ! $gpu1_busy; then
    echo "Using GPU 1."
    CUDA_VISIBLE_DEVICES=1 python train_base.py --epochs 300 --batch-size 1 --rd 34 --lr 0.0001
fi
