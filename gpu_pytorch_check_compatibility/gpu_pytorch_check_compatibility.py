"""
Shows GPU card info

sudo lshw -C display

*-display
       description: VGA compatible controller
       product: Advanced Micro Devices, Inc. [AMD/ATI]
       vendor: Advanced Micro Devices, Inc. [AMD/ATI]
       physical id: 0
       bus info: pci@0000:03:00.0
       logical name: /dev/fb0
       version: c0
       width: 64 bits
       clock: 33MHz
       capabilities: pm pciexpress msi vga_controller bus_master cap_list rom fb
       configuration: depth=32 driver=amdgpu latency=0 mode=2560x1440 resolution=2560,1440 visual=truecolor xres=2560 yres=1440
       resources: iomemory:f80-f7f iomemory:fc0-fbf irq:119 memory:f800000000-fbffffffff memory:fc00000000-fc0fffffff ioport:f000(size=256) memory:f6b00000-f6b7ffff memory:f6b80000-f6b9ffff

"""

"""
Verify Linux Distribution & Version

lsb_release -a          

No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 24.04.3 LTS
Release:        24.04
Codename:       noble
"""

"""
Verify Kernel Version
uname -r            

6.14.0-37-generic
"""

"""
Verify ROCm Installation (If already installed)
apt show rocm-libs | grep Version           

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

Version: 7.1.1.70101-38~24.04
"""

"""
Verify PyTorch Compatibility

python3 -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

ROCm available: True
Device: AMD Radeon RX 9070 XT
"""

"""
Show pytorch version
python3 -c "import torch; print(torch.__version__)"

2.9.1+rocm7.1.1.git351ff442
"""

"""
python3 -c "import torch; print(torch.cuda.get_arch_list())"
['gfx908', 'gfx90a', 'gfx942', 'gfx1030', 'gfx1100', 'gfx1101', 'gfx1200', 'gfx1201', 'gfx950', 'gfx1151', 'gfx1150']
"""

"""
python3 -c 'import torch; print(torch.cuda.is_available())'

if the output is
True
means
your PyTorch installation can successfully communicate with the ROCm driver and sees your RX 9070 XT.
"""


"""
python3 -c 'import torch; print(torch.cuda.get_device_name(0))'

if the output is
AMD Radeon RX 9070 XT
means
that your card AMD Radeon RX 9070 XT is recognized by torch
"""


"""
python3 -c 'import torch; print(torch.cuda.device_count())'

if the output is
1 
means
that your that torch recognizes 1 gpu card
"""

"""
Check Permissions
groups | grep -E 'render|video'

if the output is 
andrea adm cdrom sudo dip video plugdev users lpadmin nordvpn render

This output confirms that your permissions are correctly configured.

You have both:

    render: Allows access to the Direct Rendering Infrastructure (needed for compute/ROCm).

    video: Allows access to the display controller (needed for video output and hardware acceleration).

You do not need to run usermod.
You do not need to edit udev rules.
"""



"""
These following 2 commands are compatibility overrides. They force the ROCm software to treat your GPU as a specific hardware 
architecture version (in this case, 12.0.1), regardless of what the hardware reports itself to be.
They are equivalent. They both tell "I don't care what you think this GPU is; treat it exactly like a GFX 12.0.1 (RDNA 4) device."

1) export HSA_OVERRIDE_GFX_VERSION=12.0.1: This is a Linux Shell command. It applies to any program you run in that terminal window until you close it.

2) os.environ["HSA_OVERRIDE_GFX_VERSION"] = "12.0.1": This is a Python command. It sets the variable inside your Python script. It is useful if you want to share a script with someone and ensure it runs without them needing to configure their terminal first.

HSA: Stands for Heterogeneous System Architecture (the standard AMD uses for its compute stack).
OVERRIDE_GFX_VERSION: Tells the driver to ignore the actual "Name" or "ID" of the GPU and pretend it is the version specified.
12.0.1: This is the internal code (ISA) for RDNA 4 architecture (specifically the RX 9000 series).


if the command torch.cuda.get_device_name(0) returns AMD Radeon RX 9070 XT proves that ROCm 7.1.1 natively recognizes your GPU.
So this HSA_OVERRIDE_GFX_VERSION setup is not needed
"""

"""
GFX as "Architecture Version" (The Hardware ID)
In the context of AMD GPUs and ROCm, GFX stands for Graphics
GFX refers to the Generation of the Compute Core inside your GPU
AMD names their internal chip architectures with "GFX" numbers. This is how the software (PyTorch/ROCm) knows exactly which mathematical instructions the chip can understand.
    GFX9 (Vega): Older cards (Radeon VII, MI25).
    GFX10 (RDNA 1/2): RX 5000 / RX 6000 series.
    GFX11 (RDNA 3): RX 7000 series.
    GFX12 (RDNA 4): Your Card (RX 9070 XT).
When you see GFX1201 or 12.0.1, that is the specific code name for the silicon inside your RX 9070 XT. If PyTorch 
doesn't have a file named "gfx1201" inside its library, it won't know how to talk to your card unless you "Override" 
it to look like a version it does know. (Though as we established, ROCm 7.1.1 knows your card, so you don't need to worry about this).
"""

# The "Math" Test
import torch

# 1. Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"ROCm available: {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f"Device Count: {torch.cuda.device_count()}")
print(torch.__version__)
print(torch.cuda.get_arch_list())

# 2. Create tensors on GPU
x = torch.randn(2048, 2048, device=device)
y = torch.randn(2048, 2048, device=device)

# 3. Perform operation (Matrix Multiplication)
print("Computing matrix multiplication...")
z = torch.matmul(x, y)

# 4. Move result back to CPU to verify
print("Computation complete. Verification shape:", z.cpu().shape)
print("System is fully functional for PyTorch training.")

