import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'src'))

from tf_imgaug.sequential import Sequential
from tf_imgaug.augments import *

import numpy as np
import cv2
from skimage.transform import resize

import time

N_AUGMENTS = 15
SPEED_TEST_ITERATIONS = 100

seq = Sequential([
    SomeOf(
        (0, None),
        [
            Fliplr(1),
            Flipud(1),
            Rotate((-30, 30)),
            CropAndPad((-0.3, 0.3), pad_cval=(0, 255)),
            Translate(dict(x=(-0.2, 0.2), y=(-0.3, 0.3)))
        ]
    ),
    OneOf(
        [
            Salt(p=(0, 0.2)),
            Pepper(p=(0, 0.2)),
            Dropout(p=(0.3, 0.5))
        ]
    ),
    JpegCompression(quality=5)
], n_augments=N_AUGMENTS)


images_ph = tf.placeholder(tf.uint8, (None, None, None, 3))
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

    start = time.time()
    for i in range(SPEED_TEST_ITERATIONS):
        sess.run(_images_aug,
            feed_dict={
                images_ph: [img],
            }
        )
    print('Speed: %.4f img/s' % (N_AUGMENTS * SPEED_TEST_ITERATIONS / (time.time() - start)))
images_to_show = np.concatenate([[img], images_aug], axis=0)
keypoints_to_show = np.concatenate([kpts, keypoints_aug], axis=0)
bboxes_to_show = np.concatenate([bboxes, bboxes_aug], axis=0)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(12, 12), gridspec_kw={'wspace': 0.01, 'hspace': 0.01})

for i in range(images_to_show.shape[0]):
    _ax = ax[i // 4][i % 4]
    _ax.imshow(images_to_show[i])
    _ax.scatter(keypoints_to_show[i, :, 0], keypoints_to_show[i, :, 1], marker='x', c='g')
    _ax.plot(bboxes_to_show[i, :, [0, 2, 2, 0, 0]], bboxes_to_show[i, :, [1, 1, 3, 3, 1]], c='b')
    _ax.set_axis_off()
    _ax.set_xlim((0, images_to_show.shape[2]))
    _ax.set_ylim((0, images_to_show.shape[1]))

plt.savefig('result.jpg')



