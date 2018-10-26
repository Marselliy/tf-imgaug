import os
from distutils.core import setup

version = '0.0.1'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.md')).read()
except IOError:
    README = ''

setup(
    name='tf_imgaug',
    version=version,
    author='',
    author_email='',
    description='',
    long_description=README,
    package_dir={ '': 'src' },
    packages=[ 'tf_imgaug' ],
    keywords=[ 'tensorflow', 'imgaug', 'image augmentation', 'augmentation' ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
