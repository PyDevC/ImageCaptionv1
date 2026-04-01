#!/bin/bash

BASE_DIR="COCO"
mkdir -p $BASE_DIR/images
cd $BASE_DIR

PARALLEL_DOWNLOAD="10"

URLS=(
    "http://images.cocodataset.org/zips/train2017.zip"
    "http://images.cocodataset.org/zips/val2017.zip"
    "http://images.cocodataset.org/zips/test2017.zip"
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)

echo "Starting COCO 2017 download with aria2c..."

for url in "${URLS[@]}"; do
    if [[ $url == *"annotations"* ]]; then
        aria2c -x $PARALLEL_DOWNLOAD -s $PARALLEL_DOWNLOAD -c "$url"
    else
        aria2c -x $PARALLEL_DOWNLOAD -s $PARALLEL_DOWNLOAD -c -d "images" "$url"
    fi
done

echo "Downloads complete. Proceeding to unzip..."

unzip annotations_trainval2017.zip
unzip images/train2017.zip -d images/
unzip images/val2017.zip -d images/
unzip images/test2017.zip -d images/

echo "Dataset ready in $(pwd)"
