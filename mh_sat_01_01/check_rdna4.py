import os
import sys

# --- THE EXPERIMENT ---
# We force the driver to treat the card as Native RDNA 4 (gfx1201)
# instead of pretending to be RDNA 3 (11.0.0 / 11.0.3)
target_gfx = "12.0.1"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = target_gfx

# Standard stability flags
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"

print(f"🔬 INITIALIZING TEST FOR GFX VERSION: {target_gfx}")
print("Importing PyTorch (this might take a moment if it compiles)...")

try:
    import torch
except ImportError:
    print("❌ Error: PyTorch not installed in this environment.")
    sys.exit(1)


def run_test():
    # 1. Check Visibility
    print("\n--- Check 1: Visibility ---")
    if not torch.cuda.is_available():
        print("❌ FAIL: PyTorch cannot see the GPU.")
        print("Reason: The override might be invalid for this ROCm version.")
        return False

    device_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU Found: {device_name}")
    print(f"Current Override: {os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")

    # 2. Check Memory Allocation
    print("\n--- Check 2: Memory Allocation ---")
    try:
        x = torch.randn(2048, 2048, device="cuda")
        print("✅ Allocated 16MB tensor on VRAM.")
    except Exception as e:
        print(f"❌ FAIL: Allocation crashed.\nError: {e}")
        return False

    # 3. Check FP32 Math (This is what crashed Phase 2)
    print("\n--- Check 3: Standard Math (FP32) ---")
    print("Attempting Matrix Multiplication...")
    try:
        y = torch.matmul(x, x)
        torch.cuda.synchronize()  # Wait for GPU to actually finish
        print("✅ Success! FP32 Math works.")
    except RuntimeError as e:
        print(f"❌ FAIL: PyTorch does not have kernels for {target_gfx}.")
        print(f"Error Message: {e}")
        return False

    # 4. Check FP16 Math (This is what worked in Phase 1)
    print("\n--- Check 4: Mixed Precision (FP16) ---")
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            z = torch.matmul(x, x)
        torch.cuda.synchronize()
        print("✅ Success! FP16 Math works.")
    except Exception as e:
        print(f"❌ FAIL: FP16 crashed.\nError: {e}")
        return False

    return True


if __name__ == "__main__":
    success = run_test()
    print("\n" + "=" * 30)
    if success:
        print(f"🎉 GREAT NEWS: Your PyTorch supports {target_gfx}!")
        print("You can use '12.0.1' in all your scripts.")
    else:
        print(f"⚠️ RESULT: {target_gfx} is NOT supported yet.")
        print("You must stick to '11.0.3' and use CPU for Phase 2/3.")
    print("=" * 30)