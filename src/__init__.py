"""Package marker for src; also set default env flags."""

import os

# Avoid TensorFlow imports in transformers/pipelines when not needed.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
