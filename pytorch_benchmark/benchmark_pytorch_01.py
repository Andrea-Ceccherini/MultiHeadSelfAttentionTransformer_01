import torch
import time

# Configuration
N = 8192  # Same size as your Mojo benchmark
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_tflops(n, seconds):
    ops = 2 * n ** 3
    return (ops / seconds) / 1e12


def run_benchmark(dtype, name):
    print(f"\n--- Benchmarking {name} ---")

    # 1. Allocate Memory on GPU
    # PyTorch handles memory allocation automatically
    a = torch.ones((N, N), device=DEVICE, dtype=dtype)
    b = torch.ones((N, N), device=DEVICE, dtype=dtype)

    # 2. Warmup (Wake up the GPU clock speeds)
    for _ in range(5):
        c = torch.mm(a, b)
    torch.cuda.synchronize()

    # 3. Timed Run
    print("   Running...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    # The Math: Matrix Multiplication
    c = torch.mm(a, b)

    end_event.record()
    torch.cuda.synchronize()

    # Calculate time
    elapsed_time_ms = start_event.elapsed_time(end_event)
    seconds = elapsed_time_ms / 1000.0

    tflops = get_tflops(N, seconds)
    print(f"   Time:   {seconds:.5f} s")
    print(f"   Speed:  {tflops:.2f} TFLOPS")


if __name__ == "__main__":
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"🚀 Hardware: {props.name}")
        print(f"   Memory:   {props.total_memory / 1e9:.2f} GB")

        # Test 1: Float32 (Standard Math - Comparisons to your Mojo Code)
        run_benchmark(torch.float32, "Float32 (Standard)")

        # Test 2: Float16 (AI Accelerators - The "Special Hardware")
        # RDNA 4 cores often have 2x throughput for FP16
        run_benchmark(torch.float16, "Float16 (AI Accelerators)")

    else:
        print("❌ ROCm/HIP not detected. Please install PyTorch for ROCm.")