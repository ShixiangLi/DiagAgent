import os
import sys
import ctypes

def main():
    print("=" * 60)
    print("CUDA & PyTorch DLL/SO Diagnostics")
    print("=" * 60)
    print(f"OS Platform: {sys.platform}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    
    # 1. Print environment variables related to CUDA/Venv
    print("\n[1] Filtering environment variables for CUDA/venv references:")
    if sys.platform.startswith("win"):
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        found_any = False
        for p in path_entries:
            if any(x in p.lower() for x in ["cuda", "nvidia", ".venv", "virtualenv"]):
                print(f"  PATH: {p}")
                found_any = True
        if not found_any:
            print("  No CUDA or virtualenv paths found in PATH.")
    else:
        # Linux / Unix
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        print(f"  LD_LIBRARY_PATH: {ld_path if ld_path else '<Not Set>'}")
        ld_preload = os.environ.get("LD_PRELOAD", "")
        print(f"  LD_PRELOAD: {ld_preload if ld_preload else '<Not Set>'}")
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        for p in path_entries:
            if "cuda" in p.lower() or "nvidia" in p.lower():
                print(f"  PATH contains: {p}")

    # 2. Try loading PyTorch and initializing CUDA
    print("\n[2] Attempting to import torch and initialize CUDA:")
    try:
        import torch
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Device Count: {torch.cuda.device_count()}")
            print(f"  Current Device Name: {torch.cuda.get_device_name(0)}")
            
            # Perform a test tensor multiplication on GPU to force DLL/SO symbol resolution
            print("  Running test matrix multiplication on CUDA device...")
            x = torch.randn(5, 5, device="cuda")
            y = torch.matmul(x, x)
            print("  SUCCESS: CUDA matmul executed successfully!")
            print(f"  Output shape: {y.shape}")
        else:
            print("  WARNING: CUDA is not available via PyTorch.")
    except Exception as e:
        print(f"  ERROR: PyTorch/CUDA initialization failed!")
        print(f"  Details: {type(e).__name__}: {e}")
        
        # Suggest DLL/SO directories to check or modify
        print(f"\n[3] Diagnosis Suggestion:")
        if sys.platform.startswith("win"):
            try:
                import torch
                torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
                print(f"  It seems there is a DLL search order issue. Prioritize torch lib directory:")
                print("  - PowerShell:")
                print(f'      $env:PATH = "{torch_lib_dir};" + $env:PATH')
                print("  - Git Bash / Linux Shell:")
                print(f'      export PATH="{torch_lib_dir.replace(chr(92), "/")}:$PATH"')
            except ImportError:
                print("  torch is not installed in the current environment.")
        else:
            print("  On Linux (Ubuntu), this is typically caused by:")
            print("  1. LD_LIBRARY_PATH overriding PyTorch's bundled cuBLAS library.")
            print("     Try unsetting LD_LIBRARY_PATH and running your script again:")
            print("       export LD_LIBRARY_PATH=")
            print("  2. Driver and CUDA version mismatch (H100 requires driver version >= 525.60.13 for CUDA 12).")
            print("     Check driver status with: nvidia-smi")
            print("  3. Corrupted or mismatched PyTorch wheels. Reinstall PyTorch with explicit CUDA wheels:")
            print("       pip uninstall -y torch torchvision nvidia-cublas-cu12")
            print("       pip install torch --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    main()
