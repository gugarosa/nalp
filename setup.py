from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='nalp',
      version='2.0.3',
      description='Natural Adversarial Language Processing',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Gustavo Rosa',
      author_email='gustavo.rosa@unesp.br',
      url='https://github.com/gugarosa/nalp',
      license='Apache 2.0',
      install_requires=['coverage>=5.5',
                        'gensim>=4.1.2',
                        'matplotlib>=3.3.4',
                        'mido>=1.2.9',
                        'nltk>=3.5',
                        'pylint>=2.7.2',
                        'pytest>=6.2.2',
                        'tensorflow>=2.4.1'
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
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
