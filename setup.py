from setuptools import setup, find_packages

setup(
    name='LLMTopicLabeler',
    version='0.0.1',
    description='Automated Topic Label and model building using llm',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'scikit-learn', 'ollama'],  # List your package dependencies here
)
