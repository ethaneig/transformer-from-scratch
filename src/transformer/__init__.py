import importlib.metadata

try:
    __version__ = importlib.metadata.version("transformer")
except importlib.metadata.PackageNotFoundError:
    pass