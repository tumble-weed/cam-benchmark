mkdir dummy-folder-for-rclone
touch dummy-folder-for-rclone/1

rclone copy -P  dummy-folder-for-rclone aniketsinghresearch-gdrive:dummy-folder-for-rclone
