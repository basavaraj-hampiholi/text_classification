from setuptools import setup, find_packages

setup(
  name = 'multimodal-fusion',
  packages = find_packages(exclude=['examples']),
  version = '0.0.1',
  license='MIT',
  description = 'Sentence Classification using CNNs',
  author = 'Basavaraj Hampiholi',
  author_email = 'raaj043ofc@gmail.com',
  url = 'https://github.com/b_hampiholi/mm-fusion',
  keywords = [
    'Sentence Classification',
    'CNN'
  ],
  install_requires=[
    'torch>=1.9',
    'torchvision',
    'pytorchvideo',
    'torchvideotransforms',
    'pandas',
    'nltk'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 1 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)