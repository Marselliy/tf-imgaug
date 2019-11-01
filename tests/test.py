import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'src'))

import tensorflow as tf
from tf_imgaug.sequential import Sequential
from tf_imgaug.augments import *

import unittest

class TestAugments(unittest.TestCase):

    def _abstract_shape_test(self, aug=None):

        if aug is None:
            return

        img = tf.random.uniform((1, 64, 64, 3), 0., 1., dtype=tf.float32)
        kpts = tf.random.uniform((1, 1, 2), 0., 64., dtype=tf.float32)
        bboxes = tf.random.uniform((1, 1, 4), 0., 64., dtype=tf.float32)
        segmaps = tf.random.uniform((1, 64, 64, 2), 0., 1., dtype=tf.float32)

        if type(aug) != list:
            aug = [aug]
        img_aug, kpts_aug, bboxes_aug, segmaps_aug = Sequential(aug, n_augments=1)(img, kpts, bboxes, segmaps)

        self.assertEqual(img.shape, img_aug.shape)
        self.assertEqual(kpts.shape, kpts_aug.shape)
        self.assertEqual(bboxes.shape, bboxes_aug.shape)
        self.assertEqual(segmaps.shape, segmaps_aug.shape)

    def test_translate(self):
        self._abstract_shape_test(Translate(dict(x=(-0.1, 0.1), y=(-0.1, 0.1))))

    def test_rotate(self):
        self._abstract_shape_test(Rotate((-30, 30)))

    def test_cropandpad(self):
        self._abstract_shape_test(CropAndPad((-0.1, 0.1)))

    def test_fliplr(self):
        self._abstract_shape_test(Fliplr())

    def test_flipud(self):
        self._abstract_shape_test(Flipud())

    def test_sometimes(self):
        self._abstract_shape_test(Sometimes(0.5, Rotate((-30, 30))))

    def test_someof(self):
        self._abstract_shape_test(SomeOf((0, None), [Rotate((-30, 30)), Fliplr()]))

    def test_oneof(self):
        self._abstract_shape_test(OneOf([Rotate((-30, 30)), Fliplr()]))

    def test_jpegcompression(self):
        self._abstract_shape_test(JpegCompression(20, 80))

    def test_salt(self):
        self._abstract_shape_test(Salt())

    def test_pepper(self):
        self._abstract_shape_test(Pepper())

    def test_coarsesalt(self):
        self._abstract_shape_test(CoarseSalt())

    def test_coarsepepper(self):
        self._abstract_shape_test(CoarsePepper())

    def test_dropout(self):
        self._abstract_shape_test(Dropout())

    def test_additivegaussiannoise(self):
        self._abstract_shape_test(AdditiveGaussianNoise((0.1, 0.4)))

    def test_grayscale(self):
        self._abstract_shape_test(Grayscale((0, 1)))

    def test_add(self):
        self._abstract_shape_test(Add((-0.5, 0.5)))

    def test_multiply(self):
        self._abstract_shape_test(Multiply((0.5, 2.)))

    def test_randomresize(self):
        self._abstract_shape_test(RandomResize((0.25, 0.9)))

    def test_linearcontrast(self):
        self._abstract_shape_test(LinearContrast((0.5, 2.)))

    def test_gammacontrast(self):
        self._abstract_shape_test(GammaContrast((0., 1.)))

    def test_sigmoidcontrast(self):
        self._abstract_shape_test(SigmoidContrast(2, 10))

    def test_elastictransform(self):
        self._abstract_shape_test(ElasticTransform((0.001, 0.1)))

    def test_elasticwarp(self):
        self._abstract_shape_test(ElasticWarp((2, 5), (0.01, 0.06), 'bicubic'))

    def test_hard(self):

        aug = [
            Fliplr(0.5),
            Sometimes(0.5,
                SomeOf(
                    (1, None),
                    [
                        OneOf([
                            AdditiveGaussianNoise((0.05, 0.1), per_channel=False),
                            AdditiveGaussianNoise((0.05, 0.1), per_channel=True),
                        ]),
                        OneOf([
                            Add((-0.3, 0.3), per_channel=False),
                            Add((-0.3, 0.3), per_channel=True),
                            Multiply((0.6, 1.4), per_channel=False),
                            Multiply((0.6, 1.4), per_channel=True),
                            LinearContrast((0.5, 2.0), per_channel=False),
                            LinearContrast((0.5, 2.0), per_channel=True)
                        ]),
                        Grayscale((0.5, 1)),
                        OneOf([
                            CoarseDropout((0.1, 0.3), size_percent=(0.02, 0.06), per_channel=False),
                            CoarseDropout((0.1, 0.3), size_percent=(0.02, 0.06), per_channel=True),
                            CoarseSalt((0.1, 0.3), size_percent=(0.02, 0.06)),
                            CoarsePepper((0.1, 0.3), size_percent=(0.02, 0.06)),
                            CoarseSaltAndPepper((0.1, 0.3), size_percent=(0.02, 0.06)),
                        ]),
                        SomeOf(
                            (0, None),
                            [
                                Rotate((-45, 45)),
                                CropAndPad((-0.3, 0.3), pad_cval=(0, 255)),
                                Translate(dict(x=(-0.1, 0.1), y=(-0.1, 0.1))),
                            ]
                        ),
                        RandomResize((0.5, 0.9))
                    ],
                    random_order=True
                )
            ),
            Sometimes(0.5, JpegCompression(quality=(20, 80)))
        ]
        self._abstract_shape_test(aug)

if __name__ == '__main__':
    unittest.main()
