from setuptools import setup, find_packages

setup(
    name="pose_estimation_comparison",
    version="0.1.0",
    description="Comparative study of PVN3D, FoundationPose, and AttentionPose for 6D pose estimation under occlusion",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.7.0",
        "pybullet>=3.2.5",
        "trimesh>=3.21.0",
        "open3d>=0.17.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
        "scipy>=1.10.0",
        "pillow>=9.5.0",
        "scikit-learn>=1.2.0",
        "transforms3d>=0.4.1",
    ],
    extras_require={
        "ros": ["rclpy", "sensor_msgs", "geometry_msgs"],
    },
    python_requires=">=3.8",
)
