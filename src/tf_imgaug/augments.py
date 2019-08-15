import tensorflow as tf
import math
import random

from .utils import p_to_tensor, coarse_map

try:
    tf_image_resize = tf.image.resize
except:
    tf_image_resize = tf.image.resize_images


class AbstractAugment:

    def __init__(self, seed=1337, separable=True):
        self.random = random.Random(seed)
        self.separable = separable

    def _set_seed(self, seed):
        self.random.seed(seed)

    def _gen_seed(self):
        return self.random.randint(0, 2 ** 32)

    def _set_formats(self, keypoints_format, bboxes_format):
        self.keypoints_format = keypoints_format
        self.bboxes_format = bboxes_format

    def _augment_images(self, images):
        return images
    def _augment_keypoints(self, keypoints):
        return keypoints
    def _augment_bboxes(self, bboxes):
        return bboxes
    def _augment_segmaps(self, segmaps):
        return segmaps

    def _init_rng(self):
        pass

    def __call__(self, images, keypoints, bboxes, segmaps):
        with tf.name_scope(type(self).__name__):
            self.last_shape = tf.shape(images)
            self.last_dtype = images.dtype

            def _aug(e):
                self._init_rng()
                return (
                    self._augment_images(e[0]),
                    self._augment_keypoints(e[1]),
                    self._augment_bboxes(e[2]),
                    self._augment_segmaps(e[3])
                )

            if self.separable:
                def wrapper(args):
                    return _aug(args)

                images_aug, keypoints_aug, bboxes_aug, segmaps_aug = tf.map_fn(wrapper, tuple([tf.expand_dims(e, axis=1) for e in (images, keypoints, bboxes, segmaps)]))
                return images_aug[:, 0], keypoints_aug[:, 0], bboxes_aug[:, 0], segmaps_aug[:, 0]
            else:
                images_aug, keypoints_aug, bboxes_aug, segmaps_aug = _aug((images, keypoints, bboxes, segmaps))
                return images_aug, keypoints_aug, bboxes_aug, segmaps_aug

class Noop(AbstractAugment):

    def __init__(self):
        super(Noop, self).__init__(separable=False)


class Translate(AbstractAugment):

    def __init__(self, translate_percent, interpolation='BILINEAR'):
        super(Translate, self).__init__(separable=False)
        self.translate_percent = translate_percent
        self.interpolation = interpolation

    def _init_rng(self):
        self.trans_x = p_to_tensor(self.translate_percent['x'], shape=self.last_shape[:1], seed=self._gen_seed())
        self.trans_x = self.trans_x * tf.cast(self.last_shape[2], tf.float32)

        self.trans_y = p_to_tensor(self.translate_percent['y'], shape=self.last_shape[:1], seed=self._gen_seed())
        self.trans_y = self.trans_y * tf.cast(self.last_shape[1], tf.float32)

        self.translations_xy = tf.stack([self.trans_x, self.trans_y], axis=-1)
        self.translations_yx = tf.stack([self.trans_y, self.trans_x], axis=-1)

    def _augment_images(self, images):
        return tf.contrib.image.translate(images, self.translations_xy, self.interpolation)

    def _augment_keypoints(self, keypoints):
        if self.keypoints_format == 'xy':
            return keypoints + tf.expand_dims(self.translations_xy, axis=1)
        elif self.keypoints_format == 'yx':
            return keypoints + tf.expand_dims(self.translations_yx, axis=1)
        else:
            raise ValueError('Unsupported keypoints format: %s' % self.keypoints_format)

    def _augment_bboxes(self, bboxes):
        if self.bboxes_format == 'xyxy':
            return bboxes + tf.expand_dims(tf.tile(self.translations_xy, [1, 2]), axis=1)
        elif self.bboxes_format == 'yxyx':
            return bboxes + tf.expand_dims(tf.tile(self.translations_yx, [1, 2]), axis=1)
        elif self.bboxes_format == 'xywh':
            return bboxes + tf.expand_dims(tf.concat([self.translations_xy, tf.zeros_like(self.translations_xy)], axis=-1), axis=1)
        elif self.bboxes_format == 'yxhw':
            return bboxes + tf.expand_dims(tf.concat([self.translations_yx, tf.zeros_like(self.translations_yx)], axis=-1), axis=1)
        else:
            raise ValueError('Unsupported bboxes format: %s' % self.bboxes_format)

    def _augment_segmaps(self, segmaps):
        return tf.contrib.image.translate(segmaps, self.translations_xy, self.interpolation)


class Rotate(AbstractAugment):

    def __init__(self, rotations, interpolation='BILINEAR'):
        super(Rotate, self).__init__(separable=False)
        self.rotations = rotations
        self.interpolation = interpolation

    def _init_rng(self):
        self.angles = p_to_tensor(self.rotations, shape=self.last_shape[:1], seed=self._gen_seed()) * math.pi / 180

    def _augment_images(self, images):
        return tf.contrib.image.rotate(images, self.angles, self.interpolation)

    def _augment_keypoints(self, keypoints):
        if self.keypoints_format == 'xy':
            angles = self.angles
            angles = tf.expand_dims(-angles, axis=1)

            shape = tf.cast([self.last_shape[2], self.last_shape[1]], tf.float32)

            keypoints = keypoints - (shape / 2)
            keypoints = tf.stack([
                tf.cos(angles) * keypoints[..., 0] - tf.sin(angles) * keypoints[..., 1],
                tf.sin(angles) * keypoints[..., 0] + tf.cos(angles) * keypoints[..., 1]
            ], axis=-1) + (shape / 2)

            return keypoints
        elif self.keypoints_format == 'yx':
            angles = self.angles
            angles = tf.expand_dims(-angles, axis=1)

            shape = tf.cast([self.last_shape[1], self.last_shape[2]], tf.float32)

            keypoints = keypoints - (shape / 2)
            keypoints = tf.stack([
                tf.sin(angles) * keypoints[..., 1] + tf.cos(angles) * keypoints[..., 0],
                tf.cos(angles) * keypoints[..., 1] - tf.sin(angles) * keypoints[..., 0]
            ], axis=-1) + (shape / 2)

            return keypoints
        else:
            raise ValueError('Unsupported keypoints format: %s' % self.keypoints_format)

    def _augment_bboxes(self, bboxes):
        if self.bboxes_format == 'xyxy':
            angles = self.angles

            shape = tf.cast([self.last_shape[2], self.last_shape[1]], tf.float32)

            a = tf.cast((bboxes[..., 2] - bboxes[..., 0]) / 2, tf.float32)
            b = tf.cast((bboxes[..., 3] - bboxes[..., 1]) / 2, tf.float32)
            c_x = tf.cast((bboxes[..., 2] + bboxes[..., 0]) / 2, tf.float32)
            c_y = tf.cast((bboxes[..., 3] + bboxes[..., 1]) / 2, tf.float32)
            angles = tf.expand_dims(-angles, axis=1)

            angles = tf.cast(angles, tf.float32)
            a,b = tf.sqrt(tf.square(a * tf.cos(angles)) + tf.square(b * tf.sin(angles))), tf.sqrt(tf.square(a * tf.sin(angles)) + tf.square(b * tf.cos(angles)))

            c_x, c_y = (
                tf.cos(angles) * (c_x - shape[1] / 2) - tf.sin(angles) * (c_y - shape[0] / 2) + shape[1] / 2,
                tf.sin(angles) * (c_x - shape[1] / 2) + tf.cos(angles) * (c_y - shape[0] / 2) + shape[0] / 2
            )

            bboxes = tf.stack([
                c_x - a,
                c_y - b,
                c_x + a,
                c_y + b,
            ], axis=-1)
            return bboxes
        elif self.bboxes_format == 'yxyx':
            angles = self.angles

            shape = tf.cast([self.last_shape[1], self.last_shape[2]], tf.float32)

            a = tf.cast((bboxes[..., 3] - bboxes[..., 1]) / 2, tf.float32)
            b = tf.cast((bboxes[..., 2] - bboxes[..., 0]) / 2, tf.float32)
            c_x = tf.cast((bboxes[..., 3] + bboxes[..., 1]) / 2, tf.float32)
            c_y = tf.cast((bboxes[..., 2] + bboxes[..., 0]) / 2, tf.float32)
            angles = tf.expand_dims(-angles, axis=1)

            angles = tf.cast(angles, tf.float32)
            a,b = (tf.sqrt(tf.square(a * tf.cos(angles)) + tf.square(b * tf.sin(angles))),
                   tf.sqrt(tf.square(a * tf.sin(angles)) + tf.square(b * tf.cos(angles))))

            c_x, c_y = (
                tf.cos(angles) * (c_x - shape[1] / 2) - tf.sin(angles) * (c_y - shape[0] / 2) + shape[1] / 2,
                tf.sin(angles) * (c_x - shape[1] / 2) + tf.cos(angles) * (c_y - shape[0] / 2) + shape[0] / 2
            )

            bboxes = tf.stack([
                c_y - b,
                c_x - a,
                c_y + b,
                c_x + a,
            ], axis=-1)
            return bboxes
        elif self.bboxes_format == 'xywh':
            angles = self.angles

            shape = tf.cast([self.last_shape[2], self.last_shape[1]], tf.float32)

            a = tf.cast((bboxes[..., 2]) / 2, tf.float32)
            b = tf.cast((bboxes[..., 3]) / 2, tf.float32)
            c_x = tf.cast((bboxes[..., 0] + bboxes[..., 2] / 2), tf.float32)
            c_y = tf.cast((bboxes[..., 1] + bboxes[..., 3] / 2) , tf.float32)
            angles = tf.expand_dims(-angles, axis=1)

            angles = tf.cast(angles, tf.float32)
            a,b = tf.sqrt(tf.square(a * tf.cos(angles)) + tf.square(b * tf.sin(angles))), tf.sqrt(tf.square(a * tf.sin(angles)) + tf.square(b * tf.cos(angles)))

            c_x, c_y = (
                tf.cos(angles) * (c_x - shape[1] / 2) - tf.sin(angles) * (c_y - shape[0] / 2) + shape[1] / 2,
                tf.sin(angles) * (c_x - shape[1] / 2) + tf.cos(angles) * (c_y - shape[0] / 2) + shape[0] / 2
            )

            bboxes = tf.stack([
                c_x - a,
                c_y - b,
                a * 2,
                b * 2,
            ], axis=-1)
            return bboxes
        elif self.bboxes_format == 'yxhw':
            angles = self.angles

            shape = tf.cast([self.last_shape[1], self.last_shape[2]], tf.float32)

            a = tf.cast((bboxes[..., 3]) / 2, tf.float32)
            b = tf.cast((bboxes[..., 2]) / 2, tf.float32)
            c_x = tf.cast((bboxes[..., 1] + bboxes[..., 3] / 2), tf.float32)
            c_y = tf.cast((bboxes[..., 0] + bboxes[..., 2] / 2) , tf.float32)
            angles = tf.expand_dims(-angles, axis=1)

            angles = tf.cast(angles, tf.float32)
            a,b = (tf.sqrt(tf.square(a * tf.cos(angles)) + tf.square(b * tf.sin(angles))),
                   tf.sqrt(tf.square(a * tf.sin(angles)) + tf.square(b * tf.cos(angles))))

            c_x, c_y = (
                tf.cos(angles) * (c_x - shape[1] / 2) - tf.sin(angles) * (c_y - shape[0] / 2) + shape[1] / 2,
                tf.sin(angles) * (c_x - shape[1] / 2) + tf.cos(angles) * (c_y - shape[0] / 2) + shape[0] / 2
            )

            bboxes = tf.stack([
                c_y - b,
                c_x - a,
                b * 2,
                a * 2,
            ], axis=-1)
            return bboxes
        else:
            raise ValueError('Unsupported bboxes format: %s' % self.bboxes_format)

    def _augment_segmaps(self, segmaps):
        return tf.contrib.image.rotate(segmaps, self.angles, self.interpolation)

class CropAndPad(AbstractAugment):

    def __init__(self, percent, pad_cval=0, mode='CONSTANT', keep_ratio=False):
        super(CropAndPad, self).__init__()
        self.percent = percent
        self.pad_cval = pad_cval
        self.mode = mode
        self.keep_ratio = keep_ratio

    def _init_rng(self):
        if self.keep_ratio:
            crop_and_pads = p_to_tensor(self.percent, shape=(3,), dtype=tf.float32)
            crop_and_pads = tf.concat([
                crop_and_pads,
                tf.expand_dims(crop_and_pads[2] + crop_and_pads[0] - crop_and_pads[1], axis=0)
            ], axis=0)
        else:
            crop_and_pads = p_to_tensor(self.percent, shape=(4,), dtype=tf.float32)
        crop_and_pads = crop_and_pads * tf.cast(tf.concat([self.last_shape[1:3]] * 2, axis=0), tf.float32)
        self.crop_and_pads = tf.cast(crop_and_pads, tf.int32)

    def _augment_images(self, images):
        dtype = images.dtype
        crop_and_pads = self.crop_and_pads
        _images = images

        crops = tf.clip_by_value(crop_and_pads, tf.minimum(0, tf.reduce_min(crop_and_pads)), 0)
        images = tf.slice(
            images,
            tf.concat([[0], -crops[:2], [0]], axis=0),
            tf.concat([[-1], self.last_shape[1:3] + crops[:2] + crops[2:], [-1]], axis=0)
        )
        pads = tf.clip_by_value(crop_and_pads, 0, tf.maximum(0, tf.reduce_max(crop_and_pads)))
        pad_cval = p_to_tensor(self.pad_cval, (), dtype=images.dtype)
        images = tf.pad(images, tf.stack([[0, 0], pads[::2], pads[1::2], [0, 0]], axis=0), mode=self.mode, constant_values=pad_cval)

        resized = tf_image_resize(
            images,
            self.last_shape[1:3]
        )
        try:
            resized = tf.reshape(resized, [_images.shape[0].value, tf.shape(_images)[1], tf.shape(_images)[2], _images.shape[3].value])
        except:
            pass

        return tf.image.convert_image_dtype(resized, dtype)

    def _augment_keypoints(self, keypoints):
        if self.keypoints_format == 'xy':
            crop_and_pads = tf.cast([self.crop_and_pads[1], self.crop_and_pads[0], self.crop_and_pads[3], self.crop_and_pads[2]], tf.float32)
            _shape = tf.cast([self.last_shape[2], self.last_shape[1]], tf.float32)
            shift = tf.reshape(crop_and_pads[:2], (1, 1, 2))
            scale = tf.reshape((crop_and_pads[:2] + crop_and_pads[2:] + _shape) / _shape, (1, 1, 2))
            keypoints = (keypoints + shift) / scale

            return keypoints
        elif self.keypoints_format == 'yx':
            crop_and_pads = tf.cast([self.crop_and_pads[0], self.crop_and_pads[1], self.crop_and_pads[2], self.crop_and_pads[3]], tf.float32)
            _shape = tf.cast([self.last_shape[1], self.last_shape[2]], tf.float32)
            shift = tf.reshape(crop_and_pads[:2], (1, 1, 2))
            scale = tf.reshape((crop_and_pads[:2] + crop_and_pads[2:] + _shape) / _shape, (1, 1, 2))
            keypoints = (keypoints + shift) / scale

            return keypoints
        else:
            raise ValueError('Unsupported keypoints format: %s' % self.keypoints_format)

    def _augment_bboxes(self, bboxes):
        if self.bboxes_format == 'xyxy':
            crop_and_pads = tf.cast([self.crop_and_pads[1], self.crop_and_pads[0], self.crop_and_pads[3], self.crop_and_pads[2]], tf.float32)
            _shape = tf.cast([self.last_shape[2], self.last_shape[1]], tf.float32)
            shift = tf.reshape(crop_and_pads[:2], (1, 1, 2))
            scale = tf.reshape((crop_and_pads[:2] + crop_and_pads[2:] + _shape) / _shape, (1, 1, 2))
            bboxes = (bboxes + tf.tile(shift, (1, 1, 2))) / tf.tile(scale, (1, 1, 2))

            return bboxes
        elif self.bboxes_format == 'yxyx':
            crop_and_pads = tf.cast([self.crop_and_pads[0], self.crop_and_pads[1], self.crop_and_pads[2], self.crop_and_pads[3]], tf.float32)
            _shape = tf.cast([self.last_shape[1], self.last_shape[2]], tf.float32)
            shift = tf.reshape(crop_and_pads[:2], (1, 1, 2))
            scale = tf.reshape((crop_and_pads[:2] + crop_and_pads[2:] + _shape) / _shape, (1, 1, 2))
            bboxes = (bboxes + tf.tile(shift, (1, 1, 2))) / tf.tile(scale, (1, 1, 2))

            return bboxes
        elif self.bboxes_format == 'xywh':
            crop_and_pads = tf.cast([self.crop_and_pads[1], self.crop_and_pads[0], self.crop_and_pads[3], self.crop_and_pads[2]], tf.float32)
            _shape = tf.cast([self.last_shape[2], self.last_shape[1]], tf.float32)
            shift = tf.reshape(crop_and_pads[:2], (1, 1, 2))
            scale = tf.reshape((crop_and_pads[:2] + crop_and_pads[2:] + _shape) / _shape, (1, 1, 2))
            bboxes = (bboxes + tf.concat([shift, tf.zeros_like(shift)], axis=-1)) / tf.tile(scale, (1, 1, 2))

            return bboxes
        elif self.bboxes_format == 'yxhw':
            crop_and_pads = tf.cast([self.crop_and_pads[0], self.crop_and_pads[1], self.crop_and_pads[2], self.crop_and_pads[3]], tf.float32)
            _shape = tf.cast([self.last_shape[1], self.last_shape[2]], tf.float32)
            shift = tf.reshape(crop_and_pads[:2], (1, 1, 2))
            scale = tf.reshape((crop_and_pads[:2] + crop_and_pads[2:] + _shape) / _shape, (1, 1, 2))
            bboxes = (bboxes + tf.concat([shift, tf.zeros_like(shift)], axis=-1)) / tf.tile(scale, (1, 1, 2))

            return bboxes
        else:
            raise ValueError('Unsupported bboxes format: %s' % self.bboxes_format)

    def _augment_segmaps(self, segmaps):
        dtype = segmaps.dtype
        crop_and_pads = self.crop_and_pads
        _segmaps = segmaps

        crops = tf.clip_by_value(crop_and_pads, tf.minimum(0, tf.reduce_min(crop_and_pads)), 0)
        segmaps = tf.slice(
            segmaps,
            tf.concat([[0], -crops[:2], [0]], axis=0),
            tf.concat([[-1], self.last_shape[1:3] + crops[:2] + crops[2:], [-1]], axis=0)
        )
        pads = tf.clip_by_value(crop_and_pads, 0, tf.maximum(0, tf.reduce_max(crop_and_pads)))
        segmaps = tf.pad(segmaps, tf.stack([[0, 0], pads[::2], pads[1::2], [0, 0]], axis=0), mode=self.mode, constant_values=0)

        resized = tf_image_resize(
            segmaps,
            self.last_shape[1:3]
        )
        try:
            resized = tf.reshape(resized, [_segmaps.shape[0].value, tf.shape(_segmaps)[1], tf.shape(_segmaps)[2], _segmaps.shape[3].value])
        except:
            pass

        return tf.image.convert_image_dtype(resized, dtype)

class Fliplr(AbstractAugment):

    def __init__(self, p=0.5):
        super(Fliplr, self).__init__()
        self.p = p

    def _init_rng(self):
        self.flip = tf.random_uniform((), seed=self._gen_seed()) < self.p

    def _augment_images(self, images):
        return tf.cond(
            self.flip,
            lambda: tf.image.flip_left_right(images),
            lambda: images
        )

    def _augment_keypoints(self, keypoints):
        if self.keypoints_format == 'xy':
            w = tf.cast(self.last_shape[2], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([w - keypoints[..., 0], keypoints[..., 1]], axis=-1),
                lambda: keypoints
            )
        elif self.keypoints_format == 'yx':
            w = tf.cast(self.last_shape[2], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([keypoints[..., 0], w - keypoints[..., 1]], axis=-1),
                lambda: keypoints
            )
        else:
            raise ValueError('Unsupported keypoints format: %s' % self.keypoints_format)

    def _augment_bboxes(self, bboxes):
        if self.bboxes_format == 'xyxy':
            w = tf.cast(self.last_shape[2], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([w - bboxes[..., 2], bboxes[..., 1], w - bboxes[..., 0], bboxes[..., 3]], axis=-1),
                lambda: bboxes
            )
        elif self.bboxes_format == 'yxyx':
            w = tf.cast(self.last_shape[2], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([bboxes[..., 0], w - bboxes[..., 3], bboxes[..., 2], w - bboxes[..., 1]], axis=-1),
                lambda: bboxes
            )
        elif self.bboxes_format == 'xywh':
            w = tf.cast(self.last_shape[2], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([w - bboxes[..., 0] - bboxes[..., 2], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]], axis=-1),
                lambda: bboxes
            )
        elif self.bboxes_format == 'yxhw':
            w = tf.cast(self.last_shape[2], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([bboxes[..., 0], w - bboxes[..., 1] - bboxes[..., 3], bboxes[..., 2], bboxes[..., 3]], axis=-1),
                lambda: bboxes
            )
        else:
            raise ValueError('Unsupported bboxes format: %s' % self.bboxes_format)

    def _augment_segmaps(self, segmaps):
        return tf.cond(
            self.flip,
            lambda: tf.image.flip_left_right(segmaps),
            lambda: segmaps
        )

class Flipud(AbstractAugment):

    def __init__(self, p=0.5):
        super(Flipud, self).__init__()
        self.p = p

    def _init_rng(self):
        self.flip = tf.random_uniform((), seed=self._gen_seed()) < self.p

    def _augment_images(self, images):
        return tf.cond(
            self.flip,
            lambda: tf.image.flip_up_down(images),
            lambda: images
        )

    def _augment_keypoints(self, keypoints):
        if self.keypoints_format == 'xy':
            h = tf.cast(self.last_shape[1], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([keypoints[..., 0], h - keypoints[..., 1]], axis=-1),
                lambda: keypoints
            )
        elif self.keypoints_format == 'yx':
            h = tf.cast(self.last_shape[1], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([h - keypoints[..., 0], keypoints[..., 1]], axis=-1),
                lambda: keypoints
            )
        else:
            raise ValueError('Unsupported keypoints format: %s' % self.keypoints_format)

    def _augment_bboxes(self, bboxes):
        if self.bboxes_format == 'xyxy':
            h = tf.cast(self.last_shape[1], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([bboxes[..., 0], h - bboxes[..., 3], bboxes[..., 2], h - bboxes[..., 1]], axis=-1),
                lambda: bboxes
            )
        elif self.bboxes_format == 'yxyx':
            h = tf.cast(self.last_shape[1], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([h - bboxes[..., 2], bboxes[..., 1], h - bboxes[..., 0], bboxes[..., 3]], axis=-1),
                lambda: bboxes
            )
        elif self.bboxes_format == 'xywh':
            h = tf.cast(self.last_shape[1], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([bboxes[..., 0], h - bboxes[..., 1] - bboxes[..., 3], bboxes[..., 2], bboxes[..., 3]], axis=-1),
                lambda: bboxes
            )
        elif self.bboxes_format == 'yxhw':
            h = tf.cast(self.last_shape[1], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([h - bboxes[..., 0] - bboxes[..., 2], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]], axis=-1),
                lambda: bboxes
            )
        else:
            raise ValueError('Unsupported bboxes format: %s' % self.bboxes_format)

    def _augment_segmaps(self, segmaps):
        return tf.cond(
            self.flip,
            lambda: tf.image.flip_up_down(segmaps),
            lambda: segmaps
        )

class ElasticTransform(AbstractAugment):

    def __init__(self, displacement_stddev=0.01):
        super(ElasticTransform, self).__init__()
        self.displacement_stddev = displacement_stddev

    def _init_rng(self):
        displacement_stddevs = p_to_tensor(self.displacement_stddev, (1,), seed=self._gen_seed()) * tf.reduce_mean(tf.cast(self.last_shape[1:3], tf.float32))

        self.displacement_field = tf.random.normal(
            [1, self.last_shape[1], self.last_shape[2], 2],
            stddev=displacement_stddevs
        )
        self.displacement_field = tf.expand_dims(self.displacement_field[0], axis=0)

    def _augment_images(self, images):
        if any([e.value is None for e in images.shape]):
            raise ValueError('Images shape must be defined. Got %s' % str(images.shape))

        ret = tf.contrib.image.dense_image_warp(
            images,
            self.displacement_field
        )
        return ret

class ElasticWarp(AbstractAugment):

    def __init__(self, grid_size=4, displacement_stddev=0.05, interpolation='bicubic'):
        super(ElasticWarp, self).__init__()
        self.grid_size = grid_size
        self.displacement_stddev = displacement_stddev
        self.interpolation = interpolation

    def _init_rng(self):
        grid_sizes = p_to_tensor(self.grid_size, (2,), dtype=tf.int32, seed=self._gen_seed())
        displacement_stddevs = p_to_tensor(self.displacement_stddev, (1,), seed=self._gen_seed()) * tf.reduce_mean(tf.cast(self.last_shape[1:3], tf.float32))

        self.displacement_field_small = tf.random.normal(
            tf.stack([self.last_shape[0], grid_sizes[0], grid_sizes[1], 2], axis=0),
            stddev=displacement_stddevs
        )

        if self.interpolation == 'nearest_neighbor':
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        elif self.interpolation == 'bilinear':
            method = tf.image.ResizeMethod.BILINEAR
        elif self.interpolation == 'bicubic':
            method = tf.image.ResizeMethod.BICUBIC
        else:
            raise ValueError('Unknown interpolation: %s' % self.interpolation)

        self.displacement_field = tf_image_resize(self.displacement_field_small, self.last_shape[1:3], method=method)
        self.displacement_field = tf.expand_dims(self.displacement_field[0], axis=0)

    def _augment_images(self, images):
        if any([e.value is None for e in images.shape]):
            raise ValueError('Images shape must be defined. Got %s' % str(images.shape))

        ret = tf.contrib.image.dense_image_warp(
            images,
            self.displacement_field
        )
        return ret

    def _augment_keypoints(self, keypoints):
        if self.keypoints_format == 'xy':
            keypoints = keypoints[..., ::-1]
        elif self.keypoints_format == 'yx':
            pass
        else:
            raise ValueError('Unsupported keypoints format: %s' % self.keypoints_format)

        idx = tf.stack(tf.meshgrid(tf.range(self.last_shape[1]), tf.range(self.last_shape[2]))[::-1], axis=2)
        idx = tf.tile(tf.expand_dims(idx, axis=0), [self.last_shape[0], 1, 1, 1])
        pre_idx = tf.cast(idx, self.displacement_field.dtype) - self.displacement_field

        pre_idx = tf.reshape(pre_idx, (tf.shape(keypoints)[0], 1, -1, 2))
        keypoints = tf.expand_dims(keypoints, axis=2)
        dists = tf.reduce_sum(tf.abs(pre_idx - keypoints), axis=-1)
        coords = tf.cast(tf.argmin(dists, axis=2), tf.int32)
        xs = coords % self.last_shape[2]
        ys = coords // self.last_shape[2]

        if self.keypoints_format == 'xy':
            ax_order = [xs, ys]
        elif self.keypoints_format == 'yx':
            ax_order = [ys, xs]
        return tf.cast(tf.stack(ax_order, axis=2), keypoints.dtype)

    def _augment_bboxes(self, bboxes):
        ANCHORS_PER_EDGE = 1
        alphas = tf.linspace(0., 1., ANCHORS_PER_EDGE + 2)[:-1]
        anchors = tf.concat([
            tf.stack([alphas, tf.zeros_like(alphas)], axis=1),
            tf.stack([tf.ones_like(alphas), alphas], axis=1),
            tf.stack([1 - alphas, tf.ones_like(alphas)], axis=1),
            tf.stack([tf.zeros_like(alphas), (1 - alphas)], axis=1)
        ], axis=0)

        anchors = tf.expand_dims(tf.expand_dims(anchors, axis=0), axis=0)

        if self.bboxes_format == 'xyxy':
            l, t, r, b = [bboxes[..., e:(e + 1)] for e in range(4)]
        elif self.bboxes_format == 'yxyx':
            t, l, b, r = [bboxes[..., e:(e + 1)] for e in range(4)]
        elif self.bboxes_format == 'xywh':
            l, t, w, h = [bboxes[..., e:(e + 1)] for e in range(4)]
            r, b = l + w, t + h
        elif self.bboxes_format == 'yxhw':
            t, l, h, w = [bboxes[..., e:(e + 1)] for e in range(4)]
            r, b = l + w, t + h
        else:
            raise ValueError('Unsupported bboxes format: %s' % self.keypoints_format)

        xs = (l * anchors[..., 0] + r * (1 - anchors[..., 0]))
        ys = (t * anchors[..., 1] + b * (1 - anchors[..., 1]))

        if self.keypoints_format == 'xy':
            order = [xs, ys]
        elif self.keypoints_format == 'yx':
            order = [ys, xs]
        else:
            raise ValueError('Unsupported keypoints format: %s' % self.keypoints_format)

        anchor_points = tf.stack(order, axis=3)
        anchor_points_r = tf.reshape(anchor_points, (tf.shape(bboxes)[0], -1, 2))
        anchor_points_r = self._augment_keypoints(anchor_points_r)
        anchor_points_d = tf.reshape(anchor_points_r, tf.shape(anchor_points))

        if self.keypoints_format == 'xy':
            xs, ys = anchor_points_d[..., 0], anchor_points_d[..., 1]
        elif self.keypoints_format == 'yx':
            xs, ys = anchor_points_d[..., 1], anchor_points_d[..., 0]

        l, t, r, b = (
            tf.reduce_min(xs, axis=2),
            tf.reduce_min(ys, axis=2),
            tf.reduce_max(xs, axis=2),
            tf.reduce_max(ys, axis=2)
        )

        if self.bboxes_format == 'xyxy':
            return tf.stack([l, t, r, b], axis=2)
        elif self.bboxes_format == 'yxyx':
            return tf.stack([t, l, b, r], axis=2)
        elif self.bboxes_format == 'xywh':
            w, h = r - l, b - t
            return tf.stack([l, t, w, h], axis=2)
        elif self.bboxes_format == 'yxhw':
            w, h = r - l, b - t
            return tf.stack([t, l, h, w], axis=2)
        else:
            raise ValueError('Unsupported bboxes format: %s' % self.keypoints_format)

    def _augment_segmaps(self, segmaps):
        if any([e.value is None for e in segmaps.shape]):
            raise ValueError('Segmaps shape must be defined. Got %s' % str(segmaps.shape))

        ret = tf.contrib.image.dense_image_warp(
            segmaps,
            self.displacement_field
        )
        return ret


class Sometimes(AbstractAugment):

    def __init__(self, p, true_augment, false_augment=Noop()):
        super(Sometimes, self).__init__()
        self.p = p
        self.true_augment = true_augment
        self.false_augment = false_augment

        self.true_augment._set_seed(self._gen_seed())
        self.false_augment._set_seed(self._gen_seed())
        self.true_augment.separable = False
        self.false_augment.separable = False

    def _init_rng(self):
        self.flag = tf.random_uniform((), seed=self._gen_seed()) < self.p

    def _set_formats(self, keypoints_format, bboxes_format):
        self.keypoints_format = keypoints_format
        self.bboxes_format = bboxes_format
        self.true_augment._set_formats(keypoints_format, bboxes_format)
        self.false_augment._set_formats(keypoints_format, bboxes_format)

    def __call__(self, images, keypoints, bboxes, segmaps):

        with tf.name_scope(type(self).__name__):

            def _aug(e):
                self._init_rng()
                return tf.cond(
                    self.flag,
                    lambda: self.true_augment(*e),
                    lambda: self.false_augment(*e)
                )

            if self.separable:
                def wrapper(args):
                    return _aug(args)

                images_aug, keypoints_aug, bboxes_aug, segmaps_aug = tf.map_fn(wrapper, tuple([tf.expand_dims(e, axis=1) for e in (images, keypoints, bboxes, segmaps)]))
                return images_aug[:, 0], keypoints_aug[:, 0], bboxes_aug[:, 0], segmaps_aug[:, 0]
            else:
                return _aug((images, keypoints, bboxes, segmaps))

class SomeOf(AbstractAugment):

    def __init__(self, num, children_augments, random_order=False, separable=True):
        super(SomeOf, self).__init__(separable=separable)
        if type(num) == int:
            self.min_num = num
            self.max_num = num
        else:
            if num[0] is None:
                self.min_num = 0
            else:
                self.min_num = num[0]
            if num[1] is None:
                self.max_num = len(children_augments)
            else:
                self.max_num = num[1]
        self.children_augments = children_augments

        for aug in self.children_augments:
            aug.separable = False

        self.random_order = random_order

    def _init_rng(self):
        self.probs = tf.random.uniform((len(self.children_augments),), seed=self._gen_seed())
        self.count = tf.random.uniform((), minval=self.min_num, maxval=self.max_num + 1, dtype=tf.int32, seed=self._gen_seed())

    def _set_formats(self, keypoints_format, bboxes_format):
        self.keypoints_format = keypoints_format
        self.bboxes_format = bboxes_format
        for aug in self.children_augments:
            aug._set_formats(keypoints_format, bboxes_format)

    def __call__(self, images, keypoints, bboxes, segmaps):

        with tf.name_scope(type(self).__name__):
            def _aug(e):
                self._init_rng()
                num = tf.random_uniform((), minval=self.min_num, maxval=self.max_num + 1, dtype=tf.int32, seed=self._gen_seed())
                order = tf.random.shuffle(tf.range(len(self.children_augments)), seed=self._gen_seed())[:num]
                if not self.random_order:
                    order = tf.nn.top_k(order, k=num)[0][::-1]

                def _get_pred_fn_pairs(prev, cur_id):
                    def _(inp, aug):
                        def __():
                            return aug(*inp)
                        return __

                    return {tf.equal(cur_id, i): _(prev, aug) for i, aug in enumerate(self.children_augments)}

                def __aug(prev, cur_id):
                    pred_fn_pairs = _get_pred_fn_pairs(prev, cur_id)
                    return tf.case(pred_fn_pairs, exclusive=True)

                init = tuple([tf.identity(t) for t in e])
                result = tf.foldl(__aug, order, initializer=init, back_prop=False)
                return result

            if self.separable:
                def wrapper(args):
                    return _aug(args)

                images_aug, keypoints_aug, bboxes_aug, segmaps_aug = tf.map_fn(wrapper, tuple([tf.expand_dims(e, axis=1) for e in (images, keypoints, bboxes, segmaps)]))
                return images_aug[:, 0], keypoints_aug[:, 0], bboxes_aug[:, 0], segmaps_aug[:, 0]
            else:
                return _aug((images, keypoints, bboxes, segmaps))

class OneOf(SomeOf):

    def __init__(self, children_augments, separable=True):
        super(OneOf, self).__init__((1, 1), children_augments, separable)

class AbstractNoise(AbstractAugment):

    def __init__(self, noise_range, p, per_channel, coarse, soft, size_percent=0.01, seed=1337, separable=False):
        super(AbstractNoise, self).__init__(seed=seed, separable=separable)

        self.noise_range = noise_range
        self.p = p
        self.per_channel = per_channel
        self.coarse = coarse
        self.soft = soft
        self.size_percent = size_percent

    def _init_rng(self):
        if self.per_channel:
            noise_shape = self.last_shape
        else:
            noise_shape = tf.concat([self.last_shape[:-1], [1]], axis=0)

        p = p_to_tensor(self.p, tf.concat([noise_shape[:1], [1, 1], noise_shape[-1:]], axis=0), seed=self._gen_seed())
        if self.coarse:
            size_percent = p_to_tensor(self.size_percent, (), seed=self._gen_seed())
            if self.per_channel:
                map_shape = self.last_shape
            else:
                map_shape = tf.concat([self.last_shape[:-1], [1]], axis=0)
            self.mask = coarse_map(p, map_shape, size_percent, seed=self._gen_seed(), soft=self.soft)
        else:
            self.mask = tf.random_uniform(shape=noise_shape, seed=self._gen_seed()) < p
            self.mask = tf.cast(self.mask, tf.bool)

        self.noise = p_to_tensor(self.noise_range, noise_shape, dtype=self.last_dtype, seed=self._gen_seed())
        if self.last_dtype == tf.float32:
            self.noise = self.noise / 255

    def _augment_images(self, images):
        if self.p == 0:
            return images
        if self.p == 1:
            return tf.broadcast_to(self.noise, self.last_shape)

        if not self.per_channel:
            self.mask = tf.tile(self.mask, tf.concat([[1, 1, 1], self.last_shape[-1:]], axis=0))
            self.noise = tf.tile(self.noise, tf.concat([[1, 1, 1], self.last_shape[-1:]], axis=0))

        if self.mask.dtype == tf.bool:
            result = tf.where(self.mask, self.noise, images)
        else:
            result = self.mask * tf.cast(self.noise, self.mask.dtype) + (1 - self.mask) * tf.cast(images, self.mask.dtype)
            result = tf.cast(result, images.dtype)
        return result

class Salt(AbstractNoise):

    def __init__(self, p=0):
        super(Salt, self).__init__(noise_range=(128, 255), p=p, per_channel=False, coarse=False, soft=False)

class CoarseSalt(AbstractNoise):

    def __init__(self, p=0, size_percent=0.01, soft=False):
        super(CoarseSalt, self).__init__(noise_range=(128, 255), p=p, per_channel=False, coarse=True, soft=soft, size_percent=size_percent)

class Pepper(AbstractNoise):

    def __init__(self, p=0):
        super(Pepper, self).__init__(noise_range=(0, 128), p=p, per_channel=False, coarse=False, soft=False)

class CoarsePepper(AbstractNoise):

    def __init__(self, p=0, size_percent=0.01, soft=False):
        super(CoarsePepper, self).__init__(noise_range=(0, 128), p=p, per_channel=False, coarse=True, soft=soft, size_percent=size_percent)

class SaltAndPepper(AbstractNoise):

    def __init__(self, p=0):
        super(SaltAndPepper, self).__init__(noise_range=(0, 255), p=p, per_channel=False, coarse=False, soft=False)

class CoarseSaltAndPepper(AbstractNoise):

    def __init__(self, p=0, size_percent=0.01, soft=False):
        super(CoarseSaltAndPepper, self).__init__(noise_range=(0, 255), p=p, per_channel=False, coarse=True, soft=soft, size_percent=size_percent)

class Dropout(AbstractNoise):

    def __init__(self, p=0, per_channel=False):
        super(Dropout, self).__init__(noise_range=0, p=p, per_channel=per_channel, coarse=False, soft=False)

class CoarseDropout(AbstractNoise):

    def __init__(self, p=0, size_percent=0.01, per_channel=False, soft=False):
        super(CoarseDropout, self).__init__(noise_range=0, p=p, per_channel=per_channel, coarse=True, soft=soft, size_percent=size_percent)

class JpegCompression(AbstractAugment):

    def __init__(self, quality, seed=1337):
        super(JpegCompression, self).__init__(seed=seed, separable=True)
        self.quality = quality

    def _augment_images(self, images):
        if type(self.quality) == tuple:
            min_quality, max_quality = self.quality
        else:
            min_quality, max_quality = self.quality, self.quality + 1

        if all([not e.value is None for e in images.shape]):
            shape = images.shape
        else:
            shape = tf.shape(images)
        ret = tf.reshape(tf.image.random_jpeg_quality(images[0], min_quality, max_quality, seed=self._gen_seed()), shape)
        return ret

class AdditiveGaussianNoise(AbstractAugment):

    def __init__(self, scale, per_channel=True, seed=1337):
        super(AdditiveGaussianNoise, self).__init__(seed=seed, separable=False)
        self.scale = scale
        self.per_channel = per_channel

    def _augment_images(self, images):
        scale = p_to_tensor(self.scale, tf.concat([self.last_shape[:1], [1, 1, 1]], axis=0), seed=self._gen_seed())
        images_float = images
        if images_float.dtype != tf.float32:
            images_float = tf.image.convert_image_dtype(images_float, tf.float32)
        if self.per_channel:
            noise_shape = self.last_shape
        else:
            noise_shape = tf.concat([self.last_shape[:-1], [1]], axis=0)

        res = tf.image.convert_image_dtype(tf.clip_by_value(images_float + tf.random.normal(noise_shape, seed=self._gen_seed()) * scale, 0, 1), images.dtype)
        if res.dtype != images.dtype:
            res = tf.image.convert_image_dtype(res, images.dtype)
        return res

class Grayscale(AbstractAugment):

    def __init__(self, p, seed=1337):
        super(Grayscale, self).__init__(seed=seed, separable=False)
        self.p = p

    def _augment_images(self, images):
        if images.shape[0] is None:
            shape = tf.shape(images)[0]
        else:
            shape = int(images.shape[0])

        p = p_to_tensor(self.p, [shape, 1, 1, 1], seed=self._gen_seed())

        rgb_weights = [0.2989, 0.5870, 0.1140]

        images_float = images
        if images.dtype != tf.float32:
            images_float = tf.cast(images_float, tf.float32)
        r = \
            images_float[..., :1] * (p * (rgb_weights[0] - 1) + 1) + \
            images_float[..., 1:2] * p * rgb_weights[1] + \
            images_float[..., 2:3] * p * rgb_weights[2]
        g = \
            images_float[..., :1] * p * rgb_weights[0] + \
            images_float[..., 1:2] * (p * (rgb_weights[1] - 1) + 1) + \
            images_float[..., 2:3] * p * rgb_weights[2]
        b = \
            images_float[..., :1] * p * rgb_weights[0] + \
            images_float[..., 1:2] * p * rgb_weights[1] + \
            images_float[..., 2:3] * (p * (rgb_weights[2] - 1) + 1)

        res = tf.concat([r, g, b], axis=-1)
        if images.dtype != tf.float32:
            res = tf.cast(res, images.dtype)
        return res

class Add(AbstractAugment):

    def __init__(self, value, per_channel=False, seed=1337):
        super(Add, self).__init__(seed=seed, separable=False)

        self.value = value
        self.per_channel = per_channel

    def _augment_images(self, images):
        if images.shape[0] is None:
            shape = tf.shape(images)[0]
        else:
            shape = int(images.shape[0])
        value = p_to_tensor(self.value, [shape, 1, 1, 3 if self.per_channel else 1], seed=self._gen_seed())

        if images.dtype != tf.float32:
            value = tf.cast(value * 255., images.dtype)
            maxval = 255
        else:
            maxval = 1

        return tf.clip_by_value(images + value, 0, maxval)

class Multiply(AbstractAugment):

    def __init__(self, value, per_channel=False, seed=1337):
        super(Multiply, self).__init__(seed=seed, separable=False)

        self.value = value
        self.per_channel = per_channel

    def _augment_images(self, images):
        if images.shape[0] is None:
            shape = tf.shape(images)[0]
        else:
            shape = int(images.shape[0])
        value = p_to_tensor(self.value,[shape, 1, 1, 3 if self.per_channel else 1], seed=self._gen_seed())

        if images.dtype != tf.float32:
            maxval = 255
        else:
            maxval = 1

        res = tf.clip_by_value(images * value, 0, maxval)

        if res.dtype != images.dtype:
            res = tf.cast(res, images.dtype)

        return res

class RandomResize(AbstractAugment):

    def __init__(self, size_percent, seed=1337):
        super(RandomResize, self).__init__(seed=seed, separable=True)

        self.value = size_percent = size_percent

    def _augment_images(self, images):
        size = tf.cast(p_to_tensor(self.value, (), seed=self._gen_seed()) * tf.cast(self.last_shape[1:3], tf.float32), tf.int32)

        if all([not e.value is None for e in images.shape]):
            shape = images.shape
        else:
            shape = tf.shape(images)

        return tf.reshape(tf_image_resize(tf_image_resize(images, size), self.last_shape[1:3]), shape)

class LinearContrast(AbstractAugment):

    def __init__(self, alpha, per_channel=False, seed=1337):
        super(LinearContrast, self).__init__(seed=seed, separable=False)

        self.alpha = alpha
        self.per_channel = per_channel

    def _augment_images(self, images):
        if images.shape[0] is None:
            shape = tf.shape(images)[0]
        else:
            shape = int(images.shape[0])

        alpha = p_to_tensor(self.alpha, [shape, 1, 1, 3 if self.per_channel else 1], seed=self._gen_seed())

        return tf.clip_by_value(0.5 + alpha * (images - 0.5), 0., 1.)

class GammaContrast(AbstractAugment):

    def __init__(self, gamma, per_channel=False, seed=1337):
        super(GammaContrast, self).__init__(seed=seed, separable=False)

        self.gamma = gamma
        self.per_channel = per_channel

    def _augment_images(self, images):
        if images.shape[0] is None:
            shape = tf.shape(images)[0]
        else:
            shape = int(images.shape[0])

        gamma = p_to_tensor(self.gamma, [shape, 1, 1, 3 if self.per_channel else 1], seed=self._gen_seed())

        return tf.pow(images, gamma)

class SigmoidContrast(AbstractAugment):

    def __init__(self, gain, cutoff, per_channel=False, seed=1337):
        super(SigmoidContrast, self).__init__(seed=seed, separable=False)

        self.gain = gain
        self.cutoff = cutoff
        self.per_channel = per_channel

    def _augment_images(self, images):
        if images.shape[0] is None:
            shape = tf.shape(images)[0]
        else:
            shape = int(images.shape[0])
        shape = [shape, 1, 1, 3 if self.per_channel else 1]

        gain = p_to_tensor(self.gain, shape, seed=self._gen_seed())
        cutoff = p_to_tensor(self.cutoff, shape, seed=self._gen_seed())

        return 1 / (1 + tf.exp(gain * (cutoff - images)))

class DenormalizeColors(AbstractAugment):

    def __init__(self, points, stddev, per_channel=False, seed=1337):
        super(DenormalizeColors, self).__init__(seed=seed, separable=True)

        self.points = points
        self.stddev = stddev
        self.per_channel = per_channel

    def _augment_images(self, images):
        points =  p_to_tensor(self.points, (), dtype=tf.int32, seed=self._gen_seed())
        stddev =  p_to_tensor(self.stddev, (), dtype=tf.float32, seed=self._gen_seed())
        if images.shape[0] is None:
            shape = tf.shape(images)[0]
        else:
            shape = int(images.shape[0])
        shape = tf.concat([[shape], tf.shape(images)[1:]], axis=0)

        if self.per_channel:
            x = tf.reshape(tf.transpose(images, (0, 3, 1, 2)), (shape[0] * shape[-1], -1))
        else:
            x = tf.reshape(images, (shape[0], -1))

        pts_x = tf.concat([
            tf.zeros(shape=(tf.shape(x)[0], 1)),
            tf.random_uniform(shape=(tf.shape(x)[0], points - 2), seed=self._gen_seed()),
            tf.ones(shape=(tf.shape(x)[0], 1)),
        ], axis=1)
        pts_y = pts_x + tf.random_normal(stddev=stddev, shape=(tf.shape(x)[0], points), seed=self._gen_seed())
        pts_y = tf.clip_by_value(pts_y, 0, 1)
        train_points = tf.stack([pts_x, pts_y], axis=2)
        train_points, train_values = train_points[..., :1], train_points[..., 1:]
        query_points = tf.expand_dims(x, axis=2)

        query_values = tf.contrib.image.interpolate_spline(
            train_points=train_points,
            train_values=train_values,
            query_points=query_points,
            order=1
        )

        if self.per_channel:
            query_values = tf.transpose(tf.reshape(query_values, tf.gather(shape, [0, 3, 1, 2])), (0, 2, 3, 1))
        images_aug = tf.reshape(query_values, shape)

        return images_aug

class AddHue(AbstractAugment):

    def __init__(self, max_delta, seed=1337):
        super(AddHue, self).__init__(seed=seed, separable=True)
        self.max_delta = max_delta

    def _augment_images(self, images):
        return tf.image.random_hue(images, self.max_delta)
