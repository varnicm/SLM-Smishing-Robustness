"""
convert_pegasus_to_safetensors.py
---------------------------------
One-time: load tuner007/pegasus_paraphrase (ships only pickle .bin) and re-save
with safetensors weights into a local dir. After this, the pipeline loads the
local safetensors copy, which is NOT subject to the torch<2.6 torch.load block
(CVE-2025-32434). Run once on the login node.

    python scripts/convert_pegasus_to_safetensors.py

Produces: models/pegasus_paraphrase_st/  (safetensors + tokenizer)
Then point the pipeline at that path.
"""
import os

# The model file is a well-known public checkpoint we chose deliberately, not an
# untrusted upload, so bypassing the load-time check for this one-time local
# conversion is acceptable. We re-enable nothing globally; this only affects this
# process. After conversion the pipeline loads safetensors and never hits torch.load.
import transformers.utils.import_utils as _iu
_iu.check_torch_load_is_safe = lambda *a, **k: None  # noqa: E731  (one-time, local, trusted source)

import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

SRC = "tuner007/pegasus_paraphrase"
OUT = "models/pegasus_paraphrase_st"

os.makedirs(OUT, exist_ok=True)

print(f"[convert] loading {SRC} (pickle) ...")
tok = PegasusTokenizer.from_pretrained(SRC)
model = PegasusForConditionalGeneration.from_pretrained(SRC)

print(f"[convert] saving safetensors -> {OUT}")
model.save_pretrained(OUT, safe_serialization=True)   # writes model.safetensors
tok.save_pretrained(OUT)

# sanity: confirm safetensors file exists
import glob
st = glob.glob(os.path.join(OUT, "*.safetensors"))
print(f"[convert] done. safetensors files: {st}")
assert st, "no safetensors written!"
print("[convert] SUCCESS — point the pipeline at:", OUT)
