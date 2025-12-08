import time
import threading
import shutil
from pathlib import Path

import torch
import subprocess

GPU_MEM_BLOCK = 1024 * 1024

def get_gpu_memory_reserved(device: int = 0) -> int:
    """returns the current reserved GPU memory in MB"""
    torch.cuda.synchronize(device)
    return torch.cuda.memory_reserved(device) // GPU_MEM_BLOCK


def get_gpu_memory_allocated(device: int = 0) -> int:
    """returns the currently allocated GPU memory in MB"""
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device) // GPU_MEM_BLOCK


def get_gpu_memory_peak(device: int = 0) -> int:
    """returns the peak reserved memory since the last reset (MB)"""
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_reserved(device) // GPU_MEM_BLOCK


def reset_peak_memory(device: int = None) -> None:
    """reset pytorch peak memory tracker. call before measuring each stage."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        return
    raise "CUDA not available."



class GPUSampler:
    """Background GPU memory sampler using nvidia-smi.
    Records true memory usage including non-pytorch allocations."""
    def __init__(self, interval: int = 0.05, gpu_id: int = 0, output_file: Path = None) -> None:
        self.interval = interval
        self.gpu_id = gpu_id
        self.output_file = Path(output_file) if output_file else None
        self.running = False
        self.thread = None

    def _sample_loop(self):
        while self.running:
            mem = self._query_memory()
            time_start = time.time()

            if self.output_file:
                with open(self.output_file, "a") as f:
                    f.write(f"{time_start},{mem}\n")

            time.sleep(self.interval)

    def _query_memory(self):
        """returns GPU used_memory in MB using nvidia-smi"""
        try:
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.gpu_id)
                ]
            )
            return int(result.decode("utf-8").strip())
        except Exception:
            return -1

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def get_peak_memory(self):
        """parse peak value from sampler file."""
        if not self.output_file or not self.output_file.exists():
            return -1
        values = []
        with open(self.output_file) as f:
            for line in f:
                time_start, mem = line.strip().split(",")
                values.append(int(mem))

        return max(values) if values else -1
