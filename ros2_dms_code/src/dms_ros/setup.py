from setuptools import setup
import os
from glob import glob

package_name = 'dms_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Driver Monitoring System',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_node = dms_ros.camera_node:main',
            'preprocess_node = dms_ros.preprocess_node:main',
            'inference_node = dms_ros.inference_node:main',
            'alert_node = dms_ros.alert_node:main',
            'post_benchmark_node = dms_ros.post_benchmark_node:main',
        ],
    },
)
