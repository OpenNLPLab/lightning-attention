from setuptools import find_packages, setup

setup(
    name="lightning_attn",
    packages=find_packages(
        exclude=[
            "tests",
            "benchmarks",
        ]
    ),
    install_requires=["torch", "einops", "triton"],
    version="0.0.1",
    author="Doraemonzzz",
    include_package_data=True,
)
