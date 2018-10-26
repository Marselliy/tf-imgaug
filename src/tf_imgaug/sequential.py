import random
import tensorflow as tf

class Sequential:
    def __init__(self, augments, seed=random.randint(0, 2 ** 64), n_augments=1):
        self.augments = augments
        self.seed = seed
        self.n_augments = n_augments

    def __call__(self, images, keypoints=None, bboxes=None):
        if keypoints is None:
            keypoints = tf.zeros(tf.concat([tf.shape(images)[:1], [0, 2]], axis=0))
        if bboxes is None:
            bboxes = tf.zeros(tf.concat([tf.shape(images)[:1], [0, 4]], axis=0))
        random.seed(self.seed)
        res = (images, keypoints, bboxes)
        res = [tf.tile(e, tf.concat([[self.n_augments], tf.ones_like(tf.shape(e)[1:], dtype=tf.int32)], axis=0)) for e in res]
        for aug in self.augments:
            aug._set_seed(random.randint(0, 2 ** 64))
            res = aug(*res)

        return res

