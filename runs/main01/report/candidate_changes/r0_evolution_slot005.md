# Strict improvement of the development incumbent

- replicate 0, condition `evolution`, slot 5
- parent `e15415f45f78` -> child `a47ce4c0fb61`

**Before (dev):** mean 64.333 (total 772); balanced 33.000, cpu_heavy 89.000, mem_heavy 90.333, mixed 45.000

**After (dev):** mean 64.250 (total 771); balanced 33.000, cpu_heavy 89.000, mem_heavy 90.333, mixed 44.667

```diff
--- parent
+++ child
@@ -1,21 +1,21 @@
 def score(job_cpu, job_mem, free_cpu, free_mem):
-    """
-    Score candidate machines by combining:
-    1. Overall residual capacity reduction (Best Fit).
-    2. Minimizing the tighter remaining capacity (bottleneck exhaustion).
-    3. Balancing residual CPU and memory to prevent stranded resources.
-    """
     rem_cpu = free_cpu - job_cpu
     rem_mem = free_mem - job_mem
 
-    # Residual imbalance: penalize leaving a machine lopsided
-    imbalance = abs(rem_cpu - rem_mem)
+    # Favor machines that are fuller (higher used capacity)
+    used_cpu = 1000 - rem_cpu
+    used_mem = 1000 - rem_mem
 
-    # Tightest bottleneck remaining after placement
-    min_rem = min(rem_cpu, rem_mem)
+    # Alignment / Dot product between job demand and free capacity:
+    # A job should be placed where its dominant resource matches the machine's dominant free resource.
+    # We normalize scale by 1000.
+    alignment = (job_cpu * free_cpu + job_mem * free_mem) / 1000.0
 
-    # Total residual capacity
-    total_rem = rem_cpu + rem_mem
+    # Bottleneck tightness: measure how tightly the critical dimension fits
+    tightness = min(rem_cpu, rem_mem)
 
-    # Higher score = tighter fit on bottleneck, lower total waste, and better balance
-    return -(total_rem + 2.0 * min_rem + 0.5 * imbalance)
+    # Score:
+    # 1. Primary: Strongly favor fuller machines (used_cpu + used_mem)
+    # 2. Secondary: Good alignment of demand to available resource
+    # 3. Tertiary: Prefer smaller remaining bottleneck (tightness)
+    return (used_cpu + used_mem) * 2.0 + alignment - tightness * 1.5
```

## Model explanation / raw response

~~~~text
### Analysis

In the current heuristic, the heaviest penalties occur on `cpu_heavy` (89.000) and `mem_heavy` (90.333). 
The root cause is the `imbalance = abs(rem_cpu - rem_mem)` term coupled with `2.0 * min_rem`. In asymmetric workloads (`cpu_heavy` and `mem_heavy`), jobs systematically demand much more of one resource than the other. 

1. Penalizing residual imbalance (`abs(rem_cpu - rem_mem)`) actively forces the algorithm to avoid packing a cpu-heavy job into a machine that already has low residual CPU, or worse, spreads out skewed jobs across multiple machines to keep individual machines "balanced". But in a purely `cpu_heavy` stream, machines *cannot* be balanced; CPU is the sole bottleneck, and memory will inherently be stranded.
2. The current score prioritizes minimizing `min_rem` (the tighter residual capacity). When packing a resource-skewed job, placing it into a machine where that resource is already tight is good (Best-Fit on the bottleneck dimension), but weighting `min_rem` heavily alongside total residual capacity causes conflict with alignment.
3. Most importantly, a standard multi-dimensional bin packing strategy (like vector Best-Fit / dot-product alignment) packs items that align well with the machine's current consumed ratio, or prioritizes the Best-Fit on the dominant/bottleneck dimension without penalizing natural resource divergence.

To fix this:
- Strongly prioritize fullness (reducing `rem_cpu + rem_mem` or rewarding utilization).
- Prioritize packing the job into the dimension that is already tightest or best matches the job's demand ratio (cosine/dot-product alignment between job demand and remaining space, or minimizing residual on the critical dimension without penalizing imbalance).
- Heavily favor fuller machines (Best-Fit Decreasing behavior).

### Improved Program

```python
def score(job_cpu, job_mem, free_cpu, free_mem):
    rem_cpu = free_cpu - job_cpu
    rem_mem = free_mem - job_mem

    # Favor machines that are fuller (higher used capacity)
    used_cpu = 1000 - rem_cpu
    used_mem = 1000 - rem_mem

    # Alignment / Dot product between job demand and free capacity:
    # A job should be placed where its dominant resource matches the machine's dominant free resource.
    # We normalize scale by 1000.
    alignment = (job_cpu * free_cpu + job_mem * free_mem) / 1000.0

    # Bottleneck tightness: measure how tightly the critical dimension fits
    tightness = min(rem_cpu, rem_mem)

    # Score:
    # 1. Primary: Strongly favor fuller machines (used_cpu + used_mem)
    # 2. Secondary: Good alignment of demand to available resource
    # 3. Tertiary: Prefer smaller remaining bottleneck (tightness)
    return (used_cpu + used_mem) * 2.0 + alignment - tightness * 1.5
```
~~~~
