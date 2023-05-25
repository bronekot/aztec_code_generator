from setuptools import setup, find_packages

setup(
    name="aztec_code_generator",
    version="0.1",
    py_modules=["aztec_code_generator", "aztecfunctions", "tables", "azteccode_class"],
    install_requires=[
        "Pillow==9.5.0",
    ],
    author="Andrei Dziuba",
    author_email="aadziuba@ya.com",
    description="A tool for generating Aztec codes",
    url="https://github.com/bronekot/aztec_code_generator",
)
