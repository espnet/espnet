import os
import pathlib

# assume `ds` is already cast with Audio(decode=False)
_ = ds[0]["audio"]["path"]  # touch once → guarantees extraction

first_file = pathlib.Path(ds[0]["audio"]["path"]).resolve()
audio_cache_dir = first_file.parent

print("\n🗂  Hugging Face audio cache")
print("    first file :", first_file.name)
print("    full path  :", first_file)
print("    cache dir  :", audio_cache_dir)
print("    other files (head):")
for fname in list(audio_cache_dir.iterdir())[:5]:
    print("      ", fname.name)

# also show which cache roots are active
print("\n🔍 Cache environment variables in this run:")
for var in ("HF_DATASETS_CACHE", "HF_HOME"):
    print(f"    {var} =", os.environ.get(var))
