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
    OneOf(
        [
            OneOf(
                [
                    Fliplr(1),
                    Flipud(1)
                ]
            ),
            OneOf(
                [
                    Rotate((30, 30)),
                    Rotate((-30, 30))
                ]
            )
        ]
    )
])

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

import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=2)

ax[0].imshow(images[0].astype(np.uint8))
ax[0].scatter(keypoints[0, :, 0], keypoints[0, :, 1], marker='x', c='g')
ax[0].plot(bboxes[0, :, [0, 2, 2, 0, 0]], bboxes[0, :, [1, 1, 3, 3, 1]], c='b')
ax[1].imshow(images_aug[0].astype(np.uint8))
ax[1].scatter(keypoints_aug[0, :, 0], keypoints_aug[0, :, 1], marker='x', c='g')
ax[1].plot(bboxes_aug[0, :, [0, 2, 2, 0, 0]], bboxes_aug[0, :, [1, 1, 3, 3, 1]], c='b')
plt.savefig('1.png')



