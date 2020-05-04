from setuptools import setup, find_packages

setup(name='crown_ibp',
      version='1.0.0',
      install_requires=[
          'numpy',
          'torch',
          ],
      packages=find_packages()
)
'''
packages=['', 'verification_crown', 'verification_crown_vector_epsilon'],
package_dir={
  '': 'verification_crown', 
  'verification_crown': 'verification_crown',
  'verification_crown_vector_epsilon': 'verification_crown_vector_epsilon'
  },
'''


