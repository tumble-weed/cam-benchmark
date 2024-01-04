mkdir dummy-folder-for-rclone
touch dummy-folder-for-rclone/1

rclone copy -P aniketsinghresearch-gdrive:dummy-folder-for-rclone  dummy-folder-for-rclone 
