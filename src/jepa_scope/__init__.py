"""jepa-scope: see what a JEPA video encoder sees."""

__version__ = "0.1.0"

# Bump whenever a change alters what embedding produces for the same input (energy definition,
# preprocessing, windowing, PCA). Cached .latent files from another output version are re-embedded.
OUTPUT_VERSION = 2
