import setuptools

install_requires = ['numpy', 'numba', 'foolbox', 'matplotlib', 'Pillow']

setuptools.setup(
     name='measureDLS',  
     version='1.0',
     author='Zzh',
     author_email='zengzhihua1997@gmail.com',
     description='measureDLS',
     long_description='measureDLS',
     packages=setuptools.find_packages(),
     classifiers=[
         'Programming Language :: Python :: 3',
         'License :: OSI Approved :: MIT License',
         'Operating System :: OS Independent',
         'Intended Audience :: Science/Research',
         'Topic :: Scientific/Engineering :: Artificial Intelligence',
     ],
     install_requires=install_requires,
)
