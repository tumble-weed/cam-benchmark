dataset_dir="/root/bigfiles/dataset"
(
mkdir -p $dataset_dir/imagenet/bboxes  && \
gdown 173q2SclMLyVyZbIvGh_0d6P06gL7mMbt && \
tar -xvzf ILSVRC2012_bbox_val_v3.tgz -C $dataset_dir/imagenet/bboxes && \
 #imagenet/bboxes
#=================================================
gdown 1-5IOWkSv0VwO9WoGKW0YYDfRjMnVVvPi && \
mkdir -p $dataset_dir/imagenet/images && \
unzip imagenet_val.zip -d $dataset_dir/imagenet/images && \
mv $dataset_dir/imagenet/images/imagenet/* $dataset_dir/imagenet/images/ && \
rmdir $dataset_dir/imagenet/images/imagenet &&
rm imagenet_val.zip &&
rm ILSVRC2012_bbox_val_v3.tgz
)
#mv imagenet
