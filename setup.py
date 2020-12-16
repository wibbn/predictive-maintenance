from setuptools import setup, find_packages
def main():
    package_name = 'main'
    packages = find_packages(package_name)
    packages = list(map(lambda x: f'{package_name}/{x}', packages))

    setup(
        name=package_name,
        version='0.0.1',
        author='wibbn',
        description='Predictive Maintenance',
        package_dir={package_name: package_name},
        packages=packages,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.7',
        install_requires=[
        ],
    )

if __name__ == '__main__':
    main()