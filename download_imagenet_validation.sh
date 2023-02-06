mkdir -p imagenet/bboxes
gdown 173q2SclMLyVyZbIvGh_0d6P06gL7mMbt
tar -xvzf ILSVRC2012_bbox_val_v3.tgz -C imagenet/bboxes
#=================================================
gdown 1-5IOWkSv0VwO9WoGKW0YYDfRjMnVVvPi
mkdir -p imagenet/images
unzip imagenet_val.zip -d imagenet/images
mv imagenet/images/imagenet/* imagenet/images/
rmdir imagenet/images/imagenet
#mv imagenet
