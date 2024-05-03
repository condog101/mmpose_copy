import cv2

from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer(pose2d_weights='/home/connorscomputer/SpineSegmentation/mmpose/work_dirs/my_custom_config_2/best_AUC_epoch_20.pth',
                              pose2d='/home/connorscomputer/SpineSegmentation/mmpose/work_dirs/my_custom_config_2/my_custom_config_2.py')

# inferencer = MMPoseInferencer(pose2d_weights='/home/connorscomputer/SpineSegmentation/mmpose/res50_freihand_224x224-ff0799bc_20200914.pth',
#                               pose2d='/home/connorscomputer/SpineSegmentation/mmpose/work_dirs/my_custom_config_2/my_custom_config_2.py')

#blue hand
image_path = "/home/connorscomputer/SpineSegmentation/vertebrae_segmentation/reordered.jpg"

#non glove freihand
image_path = "/home/connorscomputer/SpineSegmentation/vertebrae_segmentation/00043273.jpg"

#glove hand
image_path = "/home/connorscomputer/Desktop/MC-hands-1M/Big set/Camera Kinect/Rendered View 15/Scene 's Collection 's Objects' States' Combination 0/1824.jpg"


image_path = "/home/connorscomputer/Desktop/finger_static/hand_scene_85.jpg"

# inferencer = MMPoseInferencer(pose2d='hand')

#
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


result_generator = inferencer(image_path, show=True)

result_generator = inferencer(image, show=True)
result = next(result_generator)