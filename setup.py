from setuptools import setup, find_packages


setup(name='nalp',
      version='1.0.1',
      description='Natural Adversarial Language Processing',
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
                        'sphinx>=1.8.5',
                        'sphinx-rtd-theme>=0.4.3'
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
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
