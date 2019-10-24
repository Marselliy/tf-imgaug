import random
import tensorflow as tf

class Sequential:
    def __init__(self, augments, seed=random.randint(0, 2 ** 32), n_augments=1, keypoints_format='xy', bboxes_format='xyxy'):
        self.augments = augments
        for aug in augments:
            aug._set_formats(keypoints_format, bboxes_format)
        self.random = random.Random(seed)
        self.n_augments = n_augments

    def __call__(self, images, keypoints=None, bboxes=None, segmaps=None, heatmaps=None):
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

                segmaps_none = False
                if segmaps is None:
                    segmaps_none = True
                    segmaps = tf.zeros(tf.concat([tf.shape(images)[:3], [0]], axis=0))

                heatmaps_none = False
                if heatmaps is None:
                    heatmaps_none = True
                    heatmaps = tf.zeros(tf.concat([tf.shape(images)[:3], [0]], axis=0))


                res = (images, keypoints, bboxes, segmaps, heatmaps)
                res = tuple([tf.tile(e, tf.concat([[self.n_augments], tf.ones_like(tf.shape(e)[1:], dtype=tf.int32)], axis=0)) for e in res])

            if images.dtype != tf.float32:
                res = (tf.image.convert_image_dtype(res[0], tf.float32),) + res[1:]

            if segmaps.dtype != tf.float32:
                res = res[:3] + (tf.image.convert_image_dtype(res[3], tf.float32),) + (res[4],)
                

            if heatmaps.dtype != tf.float32:
                res = res[:-1] + (tf.image.convert_image_dtype(res[4], tf.float32),)
                

            for aug in self.augments:
                aug._set_seed(self.random.randint(0, 2 ** 32))
                res = aug(*res)

            segmaps = res[3]
            segmaps = tf.concat([tf.ones_like(segmaps[..., :1]) * 0e-2, segmaps], axis=-1)
            segmaps = tf.one_hot(tf.argmax(segmaps, axis=-1), tf.shape(segmaps)[-1])[..., 1:]
            res = res[:3] + (segmaps,) + (res[4],)

            if images.dtype != tf.float32:
                res = (tf.image.convert_image_dtype(res[0], images.dtype),) + res[1:]

            if segmaps.dtype != tf.float32:
                res = res[:3] + (tf.image.convert_image_dtype(res[3], segmaps.dtype),) + (res[4],)

            if heatmaps.dtype != tf.float32:
                res = res[:-1] + (tf.image.convert_image_dtype(res[4], heatmaps.dtype),)

            result = [res[0]]
            if not keypoints_none:
                result.append(res[1])
            if not bboxes_none:
                result.append(res[2])
            if not segmaps_none:
                result.append(res[3])
            if not heatmaps_none:
                result.append(res[4])
            return tuple(result)

