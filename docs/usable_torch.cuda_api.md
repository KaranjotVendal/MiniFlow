PyTorch CUDA Memory APIs - Key Functions
For Memory Fragmentation Investigation
| API | Purpose | What It Shows |
|-----|---------|---------------|
| torch.cuda.memory_summary(device, abbreviated) | Full memory report | Pool fragmentation %, allocation stats |
| torch.cuda.memory_stats(device) | Detailed statistics | Segment counts, allocation counts, free/used blocks |
| torch.cuda.memory_allocated() | Current allocation | Total bytes in active use |
| torch.cuda.memory_reserved() | Cached memory | Total bytes in memory pool |
| torch.cuda.max_memory_allocated() | Peak usage | Maximum ever allocated |
For Memory Optimization
| API | Purpose | Usage |
|-----|---------|-------|
| torch.cuda.empty_cache() | Release cached memory | Call after model deletion |
| torch.cuda.synchronize() | Wait for GPU ops | Ensure timing accuracy |
| torch.cuda.memory_pool | Memory pool info | Check pool state |

---

PyTorch has a built-in visualization tool for exactly this. Use `torch.cuda.memory._record_memory_history()` to record the "State comparison." You can then export the trace and view it at pytorch.org/memory_viz. It will show you a "block map"—if it looks like a checkerboard, fragmentation is your answer.
