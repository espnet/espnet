"""The speaker-change symbol, resolved once for the whole recipe.

The symbol belongs to the trained checkpoint, not to this recipe. The
released model uses four ASCII question marks, a single Whisper BPE token,
id 25629. Every part of the recipe that reads or writes a separator takes
it from here, and conf/inference.yaml reads the same environment variable,
so the symbol the model is told about and the symbol the recipe rewrites
cannot drift apart.
"""

import os

SPEAKER_CHANGE_SYMBOL = os.environ.get("AMI_SOT_SPEAKER_CHANGE_SYMBOL", "????")
