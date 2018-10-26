import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'src', 'tf_imgaug'))

from sequential import Sequential
from augments import *
import numpy as np
import cv2
from skimage.transform import resize

seq = Sequential([
    Translate(translate_percent=dict(x=(-0.1, 0.1), y=(-0.1, 0.1))),
    Rotate((-30, 30)),
    OneOf(
        (
            CropAndPad((0.25, 0.25), pad_cval=(0, 255)),
            Rotate((30, 30)),
            CropAndPad((-0.25, -0.25), pad_cval=(0, 255))
        )
    ),
    Sometimes(0.5, Fliplr(1))
], n_augments=15)

images_ph = tf.placeholder(tf.float32, (None, None, None, 3))
keypoints_ph = tf.placeholder(tf.float32, (None, None, 2))
bboxes_ph = tf.placeholder(tf.float32, (None, None, 4))
_images_aug, _keypoints_aug, _bboxes_aug = seq(images=images_ph, keypoints=keypoints_ph, bboxes=bboxes_ph)

img = np.random.randint(low=0, high=50, size=(256, 256, 3), dtype=np.uint8)
cv2.circle(img, (128, 200), 30, (255, 0, 0), thickness=-1)
img[50:100, 150:200, 0] = 255
kpts = np.array([[
    [150, 50],
    [150, 100],
    [200, 50],
    [200, 100]
]])
bbxs = np.array([
    [
        [95, 170, 160, 230]
    ]
])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    images, images_aug, keypoints, keypoints_aug, bboxes, bboxes_aug = sess.run([
        images_ph, _images_aug, keypoints_ph, _keypoints_aug, bboxes_ph, _bboxes_aug
    ],
        feed_dict={
            images_ph: [img],
            keypoints_ph: kpts,
            bboxes_ph: bbxs
        }
    )

images_to_show = np.concatenate([[img], images_aug], axis=0)
keypoints_to_show = np.concatenate([kpts, keypoints_aug], axis=0)
bboxes_to_show = np.concatenate([bboxes, bboxes_aug], axis=0)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(12, 12), gridspec_kw={'wspace': 0.01, 'hspace': 0.01})

for i in range(images_to_show.shape[0]):
    _ax = ax[i // 4][i % 4]
    _ax.imshow(images_to_show[i].astype(np.uint8))
    _ax.scatter(keypoints_to_show[i, :, 0], keypoints_to_show[i, :, 1], marker='x', c='g')
    _ax.plot(bboxes_to_show[i, :, [0, 2, 2, 0, 0]], bboxes_to_show[i, :, [1, 1, 3, 3, 1]], c='b')
    _ax.set_axis_off()
    _ax.set_xlim((0, images_to_show.shape[2]))
    _ax.set_ylim((0, images_to_show.shape[1]))

plt.savefig('result.jpg')



