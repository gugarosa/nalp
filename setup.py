from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='nalp',
      version='1.0.1',
      description='Natural Adversarial Language Processing',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Gustavo Rosa',
      author_email='gth.rosa@uol.com.br',
      url='https://github.com/gugarosa/nalp',
      license='MIT',
      install_requires=['coverage>=4.5.2',
                        'gensim>=3.5.0',
                        'matplotlib>=3.0.0',
                        'nltk>=3.2.5',
                        'numpy>=1.13.3',
                        'pandas>=0.23.4',
                        'pylint>=1.7.4',
                        'pytest>=3.2.3',
                        'scikit-learn>=0.19.2',
                        'scipy>=1.1.0',
                        'tensorflow>=2.0.0-alpha0'
                        ],
      extras_require={
          'tests': ['coverage',
                    'pytest',
                    'pytest-pep8',
                    ],
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: Implementation :: PyPy',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
