from setuptools import setup, find_packages

setup(
  name = 'Text recognition',
  packages = find_packages(exclude=['examples']),
  version = '0.0.1',
  license='MIT',
  description = 'Sentence Classification using CNNs',
  author = 'Basavaraj Hampiholi',
  author_email = '',
  url = 'https://github.com/basavaraj-hampiholi/text_classification',
  keywords = [
    'Sentence Classification',
    'CNN'
  ],
  install_requires=[
    'torch>=1.9',
    'pandas',
    'nltk',
    'mlflow'
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
