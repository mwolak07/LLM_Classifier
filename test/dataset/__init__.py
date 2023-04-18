import warnings


# We want to ignore deprecation warnings, likely there because we cannot have the latest TensorFlow.
warnings.filterwarnings('ignore', category=DeprecationWarning)
