import random
import tensorflow as tf

class Sequential:
    def __init__(self, augments, seed=random.randint(0, 2 ** 32), n_augments=1):
        self.augments = augments
        self.seed = seed
        self.n_augments = n_augments

    def __call__(self, images, keypoints=None, bboxes=None):
        keypoints_none = False
        if keypoints is None:
            keypoints_none = True
            keypoints = tf.zeros(tf.concat([tf.shape(images)[:1], [0, 2]], axis=0))

        bboxes_none = False
        if bboxes is None:
            bboxes_none = True
            bboxes = tf.zeros(tf.concat([tf.shape(images)[:1], [0, 4]], axis=0))
        
        if isinstance(keypoints, tuple):
            keypoints, keypoints_format = keypoints
        else:
            keypoints_format = 'xy'

        if isinstance(bboxes, tuple):
            bboxes, bboxes_format = bboxes
        else:
            bboxes_format = 'xyxy'
        
        random.seed(self.seed)
        res = (images, keypoints, bboxes)
        res = [tf.tile(e, tf.concat([[self.n_augments], tf.ones_like(tf.shape(e)[1:], dtype=tf.int32)], axis=0)) for e in res]
        for aug in self.augments:
            aug._set_seed(random.randint(0, 2 ** 32))
            res = aug(*(res[0], (res[1], keypoints_format), (res[2], bboxes_format)))

        result = [res[0]]
        if not keypoints_none:
            result.append(res[1])
        if not bboxes_none:
            result.append(res[2])
        return tuple(result)

