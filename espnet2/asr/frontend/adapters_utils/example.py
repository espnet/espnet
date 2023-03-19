import s3prl.hub as hub

from add_adapters import add_adapters_wav2vec2


model = getattr(hub, "wav2vec2")()

print("before")
print(model, "\n")

# add adapters to layers 0, 1, 2
add_adapters_wav2vec2(model, adapter_down_dim=192, adapt_layers=[0, 1, 2])

print("\nafter")
print(model)
