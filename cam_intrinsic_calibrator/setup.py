from setuptools import setup, find_packages

package_name = 'cam_intrinsic_calibrator'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages("src"),
    package_dir={'':'src'},
    install_requires=[
        'setuptools', 'yaml', 'cv2', 'numpy', 'cv_bridge',
        'middleware', 'math', 'shutil', 'multiprocessing',
        'functools'
    ],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('launch/{}'.format(package_name), 
            ['launch/cam_intrinsic_calibrator.launch']),
        ('share/' + package_name + 'config', ['configs/config.yaml']),
    ],
    scripts=[
        'scripts' + '/run_cam_intrinsic_calibrator.py'
    ],
    zip_safe=True,
)