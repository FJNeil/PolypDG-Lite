import torch
from pathlib import Path

CKPT_PATH = Path(
    r"./data\stage3B_segformer_b0_loco_baseline\fold_test_C1\stage3B_results\best_model.pth"
)

print("=" * 80)
print("[CHECKPOINT PATH]")
print(CKPT_PATH)
print("=" * 80)

checkpoint = torch.load(CKPT_PATH, map_location="cpu")

print("\n[TOP-LEVEL TYPE]")
print(type(checkpoint))

if isinstance(checkpoint, dict):
    print("\n[TOP-LEVEL KEYS]")
    for key in checkpoint.keys():
        print(key)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    print("\n[STATE_DICT SOURCE]")
    print("checkpoint['model_state_dict']")

elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
    print("\n[STATE_DICT SOURCE]")
    print("checkpoint['state_dict']")

elif isinstance(checkpoint, dict) and "model" in checkpoint:
    state_dict = checkpoint["model"]
    print("\n[STATE_DICT SOURCE]")
    print("checkpoint['model']")

else:
    state_dict = checkpoint
    print("\n[STATE_DICT SOURCE]")
    print("checkpoint itself")

print("\n[NUMBER OF STATE_DICT KEYS]")
print(len(state_dict.keys()))

print("\n[FIRST 80 STATE_DICT KEYS]")
for i, key in enumerate(list(state_dict.keys())[:80]):
    value = state_dict[key]
    if hasattr(value, "shape"):
        print(f"{i:03d}: {key} | shape={tuple(value.shape)}")
    else:
        print(f"{i:03d}: {key} | type={type(value)}")

print("\n[CLASSIFIER / DECODE HEAD KEYS]")
for key, value in state_dict.items():
    if "classifier" in key or "decode_head" in key or "head" in key:
        if hasattr(value, "shape"):
            print(f"{key} | shape={tuple(value.shape)}")
        else:
            print(f"{key} | type={type(value)}")

print("\n[DONE]")