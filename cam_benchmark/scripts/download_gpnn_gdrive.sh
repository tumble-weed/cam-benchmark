#mkdir dummy-folder-for-rclone
#touch dummy-folder-for-rclone/1

rclone copyto -P aniketsinghresearch-gdrive:cam-saliency-results/pascal-simple-loadgen-n-auggpnn-mycam-75-890-vgg16.xz  /root/bigfiles/other/results-librecam/pascal-simple-loadgen-n-auggpnn-mycam-75-890-vgg16.xz
rclone copyto -P aniketsinghresearch-gdrive:cam-saliency-results/pascal-simple-loadgen-n-auggpnn-mycam-75-890-vgg16-saliency.bl2 /root/bigfiles/other/results-librecam/pascal-simple-loadgen-n-auggpnn-mycam-75-890-saliency.bl2
