#!/bin/bash

# 检查参数数量
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

DIRECTORY=$1

# 遍历文件夹并解压所有.zip文件到其所在的文件夹
find "$DIRECTORY" -type f -name "*.zip" -execdir unzip {} \;



