import random
import tensorflow as tf

class Sequential:
    def __init__(self, augments, seed=random.randint(0, 2 ** 32), n_augments=1):
        self.augments = augments
        self.random = random.Random(seed)
        self.n_augments = n_augments

    def __call__(self, images, keypoints=None, bboxes=None):
        with tf.name_scope('Sequential'):
            with tf.name_scope('prepare'):
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
                
                res = (images, keypoints, bboxes)
                res = tuple([tf.tile(e, tf.concat([[self.n_augments], tf.ones_like(tf.shape(e)[1:], dtype=tf.int32)], axis=0)) for e in res])

            if images.dtype != tf.float32:
                res = (tf.image.convert_image_dtype(res[0], tf.float32),) + res[1:]

            for aug in self.augments:
                aug._set_seed(self.random.randint(0, 2 ** 32))
                res = aug(*(res[0], (res[1], keypoints_format), (res[2], bboxes_format)))

            if images.dtype != tf.float32:
                res = (tf.image.convert_image_dtype(res[0], images.dtype),) + res[1:]

            result = [res[0]]
            if not keypoints_none:
                result.append(res[1])
            if not bboxes_none:
                result.append(res[2])
            return tuple(result)

