import tensorflow as tf
import math
import random

from .utils import p_to_tensor, coarse_map

class AbstractAugment:

    def __init__(self, seed=1337, separable=True):
        self.random = random.Random(seed)
        self.separable = separable

    def _set_seed(self, seed):
        self.random.seed(seed)

    def _gen_seed(self):
        return self.random.randint(0, 2 ** 32)

    def _augment_images(self, images):
        return images
    def _augment_keypoints(self, keypoints, fmt='xy'):
        return keypoints
    def _augment_bboxes(self, bboxes, fmt='xyxy'):
        return bboxes

    def _init_rng(self):
        pass

    def __call__(self, images, keypoints, bboxes):
        with tf.name_scope(type(self).__name__):
            self.last_shape = tf.shape(images)
            self.last_dtype = images.dtype

            if not isinstance(keypoints, tuple):
                raise ValueError('keypoints is not a tuple')

            if not isinstance(bboxes, tuple):
                raise ValueError('bboxes is not a tuple')

            keypoints, keypoints_fmt = keypoints
            bboxes, bboxes_fmt = bboxes

            def _aug(e):
                self._init_rng()
                return (
                    self._augment_images(e[0]), 
                    self._augment_keypoints(*e[1]),
                    self._augment_bboxes(*e[2])
                )

            if self.separable:
                def wrapper(args):
                    return _aug((args[0], (args[1], keypoints_fmt), (args[2], bboxes_fmt)))

                images_aug, keypoints_aug, bboxes_aug = tf.map_fn(wrapper, tuple([tf.expand_dims(e, axis=1) for e in (images, keypoints, bboxes)]))
                return images_aug[:, 0], keypoints_aug[:, 0], bboxes_aug[:, 0]
            else:
                return _aug((images, (keypoints, keypoints_fmt), (bboxes, bboxes_fmt)))

class Noop(AbstractAugment):

    def __init__(self):
        super(Noop, self).__init__(separable=False)

    def _augment_images(self, images):
        return images

    def _augment_keypoints(self, keypoints, fmt='xy'):
        return keypoints

    def _augment_bboxes(self, bboxes, fmt='xyxy'):
        return bboxes

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

    def _augment_keypoints(self, keypoints, fmt='xy'):
        if fmt == 'xy':
            return keypoints + tf.expand_dims(self.translations_xy, axis=1)
        elif fmt == 'yx':
            return keypoints + tf.expand_dims(self.translations_yx, axis=1)
        else:
            raise ValueError('Unsupported keypoints format: %s' % fmt)

    def _augment_bboxes(self, bboxes, fmt='xyxy'):
        if fmt == 'xyxy':
            return bboxes + tf.expand_dims(tf.tile(self.translations_xy, [1, 2]), axis=1)
        elif fmt == 'yxyx':
            return bboxes + tf.expand_dims(tf.tile(self.translations_yx, [1, 2]), axis=1)
        elif fmt == 'xywh':
            return bboxes + tf.expand_dims(tf.concat([self.translations_xy, tf.zeros_like(self.translations_xy)], axis=-1), axis=1)
        elif fmt == 'yxhw':
            return bboxes + tf.expand_dims(tf.concat([self.translations_yx, tf.zeros_like(self.translations_yx)], axis=-1), axis=1)
        else:
            raise ValueError('Unsupported bboxes format: %s' % fmt)


class Rotate(AbstractAugment):

    def __init__(self, rotations, interpolation='BILINEAR'):
        super(Rotate, self).__init__(separable=False)
        self.rotations = rotations
        self.interpolation = interpolation

    def _init_rng(self):
        self.angles = p_to_tensor(self.rotations, shape=self.last_shape[:1], seed=self._gen_seed()) * math.pi / 180

    def _augment_images(self, images):
        return tf.contrib.image.rotate(images, self.angles, self.interpolation)

    def _augment_keypoints(self, keypoints, fmt='xy'):
        if fmt == 'xy':
            angles = self.angles
            angles = tf.expand_dims(-angles, axis=1)

            shape = tf.cast([self.last_shape[2], self.last_shape[1]], tf.float32)

            keypoints = keypoints - (shape / 2)
            keypoints = tf.stack([
                tf.cos(angles) * keypoints[..., 0] - tf.sin(angles) * keypoints[..., 1],
                tf.sin(angles) * keypoints[..., 0] + tf.cos(angles) * keypoints[..., 1]
            ], axis=-1) + (shape / 2)

            return keypoints
        elif fmt == 'yx':
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
            raise ValueError('Unsupported keypoints format: %s' % fmt)

    def _augment_bboxes(self, bboxes, fmt='xyxy'):
        if fmt == 'xyxy':
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
        elif fmt == 'yxyx':
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
        elif fmt == 'xywh':
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
        elif fmt == 'yxhw':
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
            raise ValueError('Unsupported bboxes format: %s' % fmt)

class CropAndPad(AbstractAugment):

    def __init__(self, percent, pad_cval=0, mode='CONSTANT'):
        super(CropAndPad, self).__init__()
        self.percent = percent
        self.pad_cval = pad_cval
        self.mode = mode

    def _init_rng(self):
        crop_and_pads = p_to_tensor(self.percent, shape=(4,), dtype=tf.float32)
        crop_and_pads = crop_and_pads * tf.cast(tf.concat([self.last_shape[1:3]] * 2, axis=0), tf.float32)
        self.crop_and_pads = tf.cast(crop_and_pads, tf.int32)

    def _augment_images(self, images):
        dtype = images.dtype
        crop_and_pads = self.crop_and_pads

        crops = tf.clip_by_value(crop_and_pads, tf.minimum(0, tf.reduce_min(crop_and_pads)), 0)
        images = tf.slice(
            images,
            tf.concat([[0], -crops[:2], [0]], axis=0),
            tf.concat([[-1], self.last_shape[1:3] + crops[:2] + crops[2:], [-1]], axis=0)
        )
        pads = tf.clip_by_value(crop_and_pads, 0, tf.maximum(0, tf.reduce_max(crop_and_pads)))
        pad_cval = p_to_tensor(self.pad_cval, (), dtype=images.dtype)
        images = tf.pad(images, tf.stack([[0, 0], pads[::2], pads[1::2], [0, 0]], axis=0), mode=self.mode, constant_values=pad_cval)

        resized = tf.image.resize_images(
            images,
            self.last_shape[1:3]
        )
        return tf.image.convert_image_dtype(resized, dtype)

    def _augment_keypoints(self, keypoints, fmt='xy'):
        if fmt == 'xy':
            crop_and_pads = tf.cast([self.crop_and_pads[1], self.crop_and_pads[0], self.crop_and_pads[3], self.crop_and_pads[2]], tf.float32)
            _shape = tf.cast([self.last_shape[2], self.last_shape[1]], tf.float32)
            shift = tf.reshape(crop_and_pads[:2], (1, 1, 2))
            scale = tf.reshape((crop_and_pads[:2] + crop_and_pads[2:] + _shape) / _shape, (1, 1, 2))
            keypoints = (keypoints + shift) / scale

            return keypoints
        elif fmt == 'yx':
            crop_and_pads = tf.cast([self.crop_and_pads[0], self.crop_and_pads[1], self.crop_and_pads[2], self.crop_and_pads[3]], tf.float32)
            _shape = tf.cast([self.last_shape[1], self.last_shape[2]], tf.float32)
            shift = tf.reshape(crop_and_pads[:2], (1, 1, 2))
            scale = tf.reshape((crop_and_pads[:2] + crop_and_pads[2:] + _shape) / _shape, (1, 1, 2))
            keypoints = (keypoints + shift) / scale

            return keypoints
        else:
            raise ValueError('Unsupported keypoints format: %s' % fmt)

    def _augment_bboxes(self, bboxes, fmt='xyxy'):
        if fmt == 'xyxy':
            crop_and_pads = tf.cast([self.crop_and_pads[1], self.crop_and_pads[0], self.crop_and_pads[3], self.crop_and_pads[2]], tf.float32)
            _shape = tf.cast([self.last_shape[2], self.last_shape[1]], tf.float32)
            shift = tf.reshape(crop_and_pads[:2], (1, 1, 2))
            scale = tf.reshape((crop_and_pads[:2] + crop_and_pads[2:] + _shape) / _shape, (1, 1, 2))
            bboxes = (bboxes + tf.tile(shift, (1, 1, 2))) / tf.tile(scale, (1, 1, 2))

            return bboxes
        elif fmt == 'yxyx':
            crop_and_pads = tf.cast([self.crop_and_pads[0], self.crop_and_pads[1], self.crop_and_pads[2], self.crop_and_pads[3]], tf.float32)
            _shape = tf.cast([self.last_shape[1], self.last_shape[2]], tf.float32)
            shift = tf.reshape(crop_and_pads[:2], (1, 1, 2))
            scale = tf.reshape((crop_and_pads[:2] + crop_and_pads[2:] + _shape) / _shape, (1, 1, 2))
            bboxes = (bboxes + tf.tile(shift, (1, 1, 2))) / tf.tile(scale, (1, 1, 2))

            return bboxes
        elif fmt == 'xywh':
            crop_and_pads = tf.cast([self.crop_and_pads[1], self.crop_and_pads[0], self.crop_and_pads[3], self.crop_and_pads[2]], tf.float32)
            _shape = tf.cast([self.last_shape[2], self.last_shape[1]], tf.float32)
            shift = tf.reshape(crop_and_pads[:2], (1, 1, 2))
            scale = tf.reshape((crop_and_pads[:2] + crop_and_pads[2:] + _shape) / _shape, (1, 1, 2))
            bboxes = (bboxes + tf.concat([shift, tf.zeros_like(shift)], axis=-1)) / tf.tile(scale, (1, 1, 2))

            return bboxes
        elif fmt == 'yxhw':
            crop_and_pads = tf.cast([self.crop_and_pads[0], self.crop_and_pads[1], self.crop_and_pads[2], self.crop_and_pads[3]], tf.float32)
            _shape = tf.cast([self.last_shape[1], self.last_shape[2]], tf.float32)
            shift = tf.reshape(crop_and_pads[:2], (1, 1, 2))
            scale = tf.reshape((crop_and_pads[:2] + crop_and_pads[2:] + _shape) / _shape, (1, 1, 2))
            bboxes = (bboxes + tf.concat([shift, tf.zeros_like(shift)], axis=-1)) / tf.tile(scale, (1, 1, 2))

            return bboxes
        else:
            raise ValueError('Unsupported bboxes format: %s' % fmt)

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

    def _augment_keypoints(self, keypoints, fmt='xy'):
        if fmt == 'xy':
            w = tf.cast(self.last_shape[2], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([w - keypoints[..., 0], keypoints[..., 1]], axis=-1),
                lambda: keypoints
            )
        elif fmt == 'yx':
            w = tf.cast(self.last_shape[2], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([keypoints[..., 0], w - keypoints[..., 1]], axis=-1),
                lambda: keypoints
            )
        else:
            raise ValueError('Unsupported keypoints format: %s' % fmt)

    def _augment_bboxes(self, bboxes, fmt='xyxy'):
        if fmt == 'xyxy':
            w = tf.cast(self.last_shape[2], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([w - bboxes[..., 2], bboxes[..., 1], w - bboxes[..., 0], bboxes[..., 3]], axis=-1),
                lambda: bboxes
            )
        elif fmt == 'yxyx':
            w = tf.cast(self.last_shape[2], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([bboxes[..., 0], w - bboxes[..., 3], bboxes[..., 2], w - bboxes[..., 1]], axis=-1),
                lambda: bboxes
            )
        elif fmt == 'xywh':
            w = tf.cast(self.last_shape[2], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([w - bboxes[..., 0] - bboxes[..., 2], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]], axis=-1),
                lambda: bboxes
            )
        elif fmt == 'yxhw':
            w = tf.cast(self.last_shape[2], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([bboxes[..., 0], w - bboxes[..., 1] - bboxes[..., 3], bboxes[..., 2], bboxes[..., 3]], axis=-1),
                lambda: bboxes
            )
        else:
            raise ValueError('Unsupported bboxes format: %s' % fmt)

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

    def _augment_keypoints(self, keypoints, fmt='xy'):
        if fmt == 'xy':
            h = tf.cast(self.last_shape[1], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([keypoints[..., 0], h - keypoints[..., 1]], axis=-1),
                lambda: keypoints
            )
        elif fmt == 'yx':
            h = tf.cast(self.last_shape[1], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([h - keypoints[..., 0], keypoints[..., 1]], axis=-1),
                lambda: keypoints
            )
        else:
            raise ValueError('Unsupported keypoints format: %s' % fmt)

    def _augment_bboxes(self, bboxes, fmt='xyxy'):
        if fmt == 'xyxy':
            h = tf.cast(self.last_shape[1], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([bboxes[..., 0], h - bboxes[..., 3], bboxes[..., 2], h - bboxes[..., 1]], axis=-1),
                lambda: bboxes
            )
        elif fmt == 'yxyx':
            h = tf.cast(self.last_shape[1], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([h - bboxes[..., 2], bboxes[..., 1], h - bboxes[..., 0], bboxes[..., 3]], axis=-1),
                lambda: bboxes
            )
        elif fmt == 'xywh':
            h = tf.cast(self.last_shape[1], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([bboxes[..., 0], h - bboxes[..., 1] - bboxes[..., 3], bboxes[..., 2], bboxes[..., 3]], axis=-1),
                lambda: bboxes
            )
        elif fmt == 'yxhw':
            h = tf.cast(self.last_shape[1], tf.float32)
            return tf.cond(
                self.flip,
                lambda: tf.stack([h - bboxes[..., 0] - bboxes[..., 2], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]], axis=-1),
                lambda: bboxes
            )
        else:
            raise ValueError('Unsupported bboxes format: %s' % fmt)


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

    def __call__(self, images, keypoints, bboxes):
        if not isinstance(keypoints, tuple):
            raise ValueError('keypoints is not a tuple')

        if not isinstance(bboxes, tuple):
            raise ValueError('bboxes is not a tuple')

        with tf.name_scope(type(self).__name__):
            keypoints, keypoints_fmt = keypoints
            bboxes, bboxes_fmt = bboxes

            def _aug(e):
                self._init_rng()
                return tf.cond(
                    self.flag,
                    lambda: self.true_augment(*e),
                    lambda: self.false_augment(*e)
                )

            if self.separable:
                def wrapper(args):
                    return _aug((args[0], (args[1], keypoints_fmt), (args[2], bboxes_fmt)))

                images_aug, keypoints_aug, bboxes_aug = tf.map_fn(wrapper, tuple([tf.expand_dims(e, axis=1) for e in (images, keypoints, bboxes)]))
                return images_aug[:, 0], keypoints_aug[:, 0], bboxes_aug[:, 0]
            else:
                return _aug((images, (keypoints, keypoints_fmt), (bboxes, bboxes_fmt)))

class SomeOf(AbstractAugment):

    def __init__(self, num, children_augments, separable=True):
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

    def _init_rng(self):
        self.probs = tf.random_uniform((len(self.children_augments),), seed=self._gen_seed())
        self.count = tf.random_uniform((), minval=self.min_num, maxval=self.max_num + 1, dtype=tf.int32, seed=self._gen_seed())

    def __call__(self, images, keypoints, bboxes):
        if not isinstance(keypoints, tuple):
            raise ValueError('keypoints is not a tuple')

        if not isinstance(bboxes, tuple):
            raise ValueError('bboxes is not a tuple')

        with tf.name_scope(type(self).__name__):
            keypoints, keypoints_fmt = keypoints
            bboxes, bboxes_fmt = bboxes
            def _aug(e):
                self._init_rng()
                values, _ = tf.nn.top_k(self.probs, self.count)
                mask = tf.greater_equal(self.probs, tf.reduce_min(values))

                result = Noop()(*e)
                for i, augment in enumerate(self.children_augments):
                    augment._set_seed(random.randint(0, 2 ** 32))
                    result = tf.cond(
                        mask[i],
                        lambda: augment(*(result[0], (result[1], keypoints_fmt), (result[2], bboxes_fmt))),
                        lambda: result
                    )
                return result

            if self.separable:
                def wrapper(args):
                    return _aug((args[0], (args[1], keypoints_fmt), (args[2], bboxes_fmt)))

                images_aug, keypoints_aug, bboxes_aug = tf.map_fn(wrapper, tuple([tf.expand_dims(e, axis=1) for e in (images, keypoints, bboxes)]))
                return images_aug[:, 0], keypoints_aug[:, 0], bboxes_aug[:, 0]
            else:
                return _aug((images, (keypoints, keypoints_fmt), (bboxes, bboxes_fmt)))

class OneOf(SomeOf):

    def __init__(self, children_augments, separable=True):
        super(OneOf, self).__init__((1, 1), children_augments, separable)

class AbstractNoise(AbstractAugment):

    def __init__(self, noise_range, p, per_channel, coarse, size_percent=0.01, seed=1337, separable=False):
        super(AbstractNoise, self).__init__(seed=seed, separable=separable)

        self.noise_range = noise_range
        self.p = p
        self.per_channel = per_channel
        self.coarse = coarse
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
            self.mask = coarse_map(p, map_shape, size_percent, seed=self._gen_seed())
        else:
            self.mask = tf.random_uniform(shape=noise_shape, seed=self._gen_seed()) < p
            self.mask = tf.cast(self.mask, tf.bool)

        self.noise = p_to_tensor(self.noise_range, noise_shape, dtype=self.last_dtype, seed=self._gen_seed())

    def _augment_images(self, images):
        if self.p == 0:
            return images
        if self.p == 1:
            return tf.broadcast_to(self.noise, self.last_shape)

        if not self.per_channel:
            self.mask = tf.tile(self.mask, tf.concat([[1, 1, 1], self.last_shape[-1:]], axis=0))
            self.noise = tf.tile(self.noise, tf.concat([[1, 1, 1], self.last_shape[-1:]], axis=0))
        return tf.where(tf.cast(self.mask, tf.bool), self.noise, images)

class Salt(AbstractNoise):

    def __init__(self, p=0):
        super(Salt, self).__init__(noise_range=(128, 255), p=p, per_channel=False, coarse=False)

class CoarseSalt(AbstractNoise):

    def __init__(self, p=0, size_percent=0.01):
        super(CoarseSalt, self).__init__(noise_range=(128, 255), p=p, per_channel=False, coarse=True, size_percent=size_percent)

class Pepper(AbstractNoise):

    def __init__(self, p=0):
        super(Pepper, self).__init__(noise_range=(0, 128), p=p, per_channel=False, coarse=False)

class CoarsePepper(AbstractNoise):

    def __init__(self, p=0, size_percent=0.01):
        super(CoarsePepper, self).__init__(noise_range=(0, 128), p=p, per_channel=False, coarse=True, size_percent=size_percent)

class SaltAndPepper(AbstractNoise):

    def __init__(self, p=0):
        super(SaltAndPepper, self).__init__(noise_range=(0, 255), p=p, per_channel=False, coarse=False)

class CoarseSaltAndPepper(AbstractNoise):

    def __init__(self, p=0, size_percent=0.01):
        super(CoarseSaltAndPepper, self).__init__(noise_range=(0, 255), p=p, per_channel=False, coarse=True, size_percent=size_percent)

class Dropout(AbstractNoise):

    def __init__(self, p=0, per_channel=False):
        super(Dropout, self).__init__(noise_range=0, p=p, per_channel=per_channel, coarse=False)

class CoarseDropout(AbstractNoise):

    def __init__(self, p=0, size_percent=0.01, per_channel=False):
        super(CoarseDropout, self).__init__(noise_range=0, p=p, per_channel=per_channel, coarse=True, size_percent=size_percent)

class JpegCompression(AbstractAugment):

    def __init__(self, quality, seed=1337):
        super(JpegCompression, self).__init__(seed=seed, separable=True)
        self.quality = quality

    def _augment_images(self, images):
        if type(self.quality) == tuple:
            min_quality, max_quality = self.quality
        else:
            min_quality, max_quality = self.quality, self.quality + 1

        return tf.expand_dims(tf.image.random_jpeg_quality(images[0], min_quality, max_quality, seed=self._gen_seed()), axis=0)

class AdditiveGaussianNoise(AbstractAugment):

    def __init__(self, scale, per_channel=True, seed=1337):
        super(AdditiveGaussianNoise, self).__init__(seed=seed, separable=False)
        self.scale = scale
        self.per_channel = per_channel

    def _augment_images(self, images):
        scale = p_to_tensor(self.scale, tf.concat([self.last_shape[:1], [1, 1, 1]], axis=0), seed=self._gen_seed())
        if images.dtype != tf.float32:
            scale = scale * 255.
            maxval = 255
        else:
            maxval = 1
        if self.per_channel:
            noise_shape = self.last_shape
        else:
            noise_shape = tf.concat([self.last_shape[:-1], [1]], axis=0)

        return tf.cast(tf.clip_by_value(tf.cast(images, tf.int32) + tf.cast(tf.random_normal(noise_shape, self._gen_seed()) * scale, tf.int32), 0, maxval), images.dtype)

class Grayscale(AbstractAugment):

    def __init__(self, p, seed=1337):
        super(Grayscale, self).__init__(seed=seed, separable=False)
        self.p = p

    def _augment_images(self, images):
        p = p_to_tensor(self.p, tf.concat([self.last_shape[:1], [1, 1, 1]], axis=0), seed=self._gen_seed())
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
        value = p_to_tensor(self.value, tf.concat([self.last_shape[:1], [1, 1], [3] if self.per_channel else [1]], axis=0), seed=self._gen_seed())

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
        value = p_to_tensor(self.value, tf.concat([self.last_shape[:1], [1, 1], [3] if self.per_channel else [1]], axis=0), seed=self._gen_seed())

        if images.dtype != tf.float32:
            maxval = 255
        else:
            maxval = 1

        res = tf.clip_by_value(images * value, 0, maxval)

        if res.dtype != images.dtype:
            res = tf.cast(res, images.dtype)

        return res

