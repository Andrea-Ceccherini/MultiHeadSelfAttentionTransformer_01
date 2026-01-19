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
/opt/rocm/bin/rocminfo | grep gfx

if the output is
  Name:                    gfx1201                            
      Name:                    amdgcn-amd-amdhsa--gfx1201         
      Name:                    amdgcn-amd-amdhsa--gfx12-generic 
      
Means tha the GPU architecture is gfx1201 that is the architecture of GPU AMD Radeon RX 9070 XT     
"""

"""
/opt/rocm/bin/rocminfo
ROCk module version 6.16.6 is loaded
=====================    
HSA System Attributes    
=====================    
Runtime Version:         1.18
Runtime Ext Version:     1.14
System Timestamp Freq.:  1000.000000MHz
Sig. Max Wait Duration:  18446744073709551615 (0xFFFFFFFFFFFFFFFF) (timestamp count)
Machine Model:           LARGE                              
System Endianness:       LITTLE                             
Mwaitx:                  DISABLED
XNACK enabled:           NO
DMAbuf Support:          YES
VMM Support:             YES

==========               
HSA Agents               
==========               
*******                  
Agent 1                  
*******                  
  Name:                    AMD Ryzen 7 7800X3D 8-Core Processor
  Uuid:                    CPU-XX                             
  Marketing Name:          AMD Ryzen 7 7800X3D 8-Core Processor
  Vendor Name:             CPU                                
  Feature:                 None specified                     
  Profile:                 FULL_PROFILE                       
  Float Round Mode:        NEAR                               
  Max Queue Number:        0(0x0)                             
  Queue Min Size:          0(0x0)                             
  Queue Max Size:          0(0x0)                             
  Queue Type:              MULTI                              
  Node:                    0                                  
  Device Type:             CPU                                
  Cache Info:              
    L1:                      32768(0x8000) KB                   
  Chip ID:                 0(0x0)                             
  ASIC Revision:           0(0x0)                             
  Cacheline Size:          64(0x40)                           
  Max Clock Freq. (MHz):   5053                               
  BDFID:                   0                                  
  Internal Node ID:        0                                  
  Compute Unit:            16                                 
  SIMDs per CU:            0                                  
  Shader Engines:          0                                  
  Shader Arrs. per Eng.:   0                                  
  WatchPts on Addr. Ranges:1                                  
  Memory Properties:       
  Features:                None
  Pool Info:               
    Pool 1                   
      Segment:                 GLOBAL; FLAGS: FINE GRAINED        
      Size:                    32459836(0x1ef4c3c) KB             
      Allocatable:             TRUE                               
      Alloc Granule:           4KB                                
      Alloc Recommended Granule:4KB                                
      Alloc Alignment:         4KB                                
      Accessible by all:       TRUE                               
    Pool 2                   
      Segment:                 GLOBAL; FLAGS: EXTENDED FINE GRAINED
      Size:                    32459836(0x1ef4c3c) KB             
      Allocatable:             TRUE                               
      Alloc Granule:           4KB                                
      Alloc Recommended Granule:4KB                                
      Alloc Alignment:         4KB                                
      Accessible by all:       TRUE                               
    Pool 3                   
      Segment:                 GLOBAL; FLAGS: KERNARG, FINE GRAINED
      Size:                    32459836(0x1ef4c3c) KB             
      Allocatable:             TRUE                               
      Alloc Granule:           4KB                                
      Alloc Recommended Granule:4KB                                
      Alloc Alignment:         4KB                                
      Accessible by all:       TRUE                               
    Pool 4                   
      Segment:                 GLOBAL; FLAGS: COARSE GRAINED      
      Size:                    32459836(0x1ef4c3c) KB             
      Allocatable:             TRUE                               
      Alloc Granule:           4KB                                
      Alloc Recommended Granule:4KB                                
      Alloc Alignment:         4KB                                
      Accessible by all:       TRUE                               
  ISA Info:                
*******                  
Agent 2                  
*******                  
  Name:                    gfx1201                            
  Uuid:                    GPU-cbf523963b385e34               
  Marketing Name:          AMD Radeon RX 9070 XT              
  Vendor Name:             AMD                                
  Feature:                 KERNEL_DISPATCH                    
  Profile:                 BASE_PROFILE                       
  Float Round Mode:        NEAR                               
  Max Queue Number:        128(0x80)                          
  Queue Min Size:          64(0x40)                           
  Queue Max Size:          131072(0x20000)                    
  Queue Type:              MULTI                              
  Node:                    1                                  
  Device Type:             GPU                                
  Cache Info:              
    L1:                      32(0x20) KB                        
    L2:                      8192(0x2000) KB                    
    L3:                      65536(0x10000) KB                  
  Chip ID:                 30032(0x7550)                      
  ASIC Revision:           1(0x1)                             
  Cacheline Size:          256(0x100)                         
  Max Clock Freq. (MHz):   2400                               
  BDFID:                   768                                
  Internal Node ID:        1                                  
  Compute Unit:            64                                 
  SIMDs per CU:            2                                  
  Shader Engines:          4                                  
  Shader Arrs. per Eng.:   2                                  
  WatchPts on Addr. Ranges:4                                  
  Coherent Host Access:    FALSE                              
  Memory Properties:       
  Features:                KERNEL_DISPATCH 
  Fast F16 Operation:      TRUE                               
  Wavefront Size:          32(0x20)                           
  Workgroup Max Size:      1024(0x400)                        
  Workgroup Max Size per Dimension:
    x                        1024(0x400)                        
    y                        1024(0x400)                        
    z                        1024(0x400)                        
  Max Waves Per CU:        32(0x20)                           
  Max Work-item Per CU:    1024(0x400)                        
  Grid Max Size:           4294967295(0xffffffff)             
  Grid Max Size per Dimension:
    x                        2147483647(0x7fffffff)             
    y                        65535(0xffff)                      
    z                        65535(0xffff)                      
  Max fbarriers/Workgrp:   32                                 
  Packet Processor uCode:: 108                                
  SDMA engine uCode::      662                                
  IOMMU Support::          None                               
  Pool Info:               
    Pool 1                   
      Segment:                 GLOBAL; FLAGS: COARSE GRAINED      
      Size:                    16695296(0xfec000) KB              
      Allocatable:             TRUE                               
      Alloc Granule:           4KB                                
      Alloc Recommended Granule:2048KB                             
      Alloc Alignment:         4KB                                
      Accessible by all:       FALSE                              
    Pool 2                   
      Segment:                 GROUP                              
      Size:                    64(0x40) KB                        
      Allocatable:             FALSE                              
      Alloc Granule:           0KB                                
      Alloc Recommended Granule:0KB                                
      Alloc Alignment:         0KB                                
      Accessible by all:       FALSE                              
  ISA Info:                
    ISA 1                    
      Name:                    amdgcn-amd-amdhsa--gfx1201         
      Machine Models:          HSA_MACHINE_MODEL_LARGE            
      Profiles:                HSA_PROFILE_BASE                   
      Default Rounding Mode:   NEAR                               
      Default Rounding Mode:   NEAR                               
      Fast f16:                TRUE                               
      Workgroup Max Size:      1024(0x400)                        
      Workgroup Max Size per Dimension:
        x                        1024(0x400)                        
        y                        1024(0x400)                        
        z                        1024(0x400)                        
      Grid Max Size:           4294967295(0xffffffff)             
      Grid Max Size per Dimension:
        x                        2147483647(0x7fffffff)             
        y                        65535(0xffff)                      
        z                        65535(0xffff)                      
      FBarrier Max Size:       32                                 
    ISA 2                    
      Name:                    amdgcn-amd-amdhsa--gfx12-generic   
      Machine Models:          HSA_MACHINE_MODEL_LARGE            
      Profiles:                HSA_PROFILE_BASE                   
      Default Rounding Mode:   NEAR                               
      Default Rounding Mode:   NEAR                               
      Fast f16:                TRUE                               
      Workgroup Max Size:      1024(0x400)                        
      Workgroup Max Size per Dimension:
        x                        1024(0x400)                        
        y                        1024(0x400)                        
        z                        1024(0x400)                        
      Grid Max Size:           4294967295(0xffffffff)             
      Grid Max Size per Dimension:
        x                        2147483647(0x7fffffff)             
        y                        65535(0xffff)                      
        z                        65535(0xffff)                      
      FBarrier Max Size:       32                                 
*** Done ***             

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

