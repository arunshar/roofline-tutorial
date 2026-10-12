# How the Roofline Model Works

By [Arun Sharma](https://arunshar.com) and [Claude Code](https://claude.com/claude-code)

The roofline model is an upper bound on the speed of a kernel. It treats the chip as two
resources, arithmetic units and a path to main memory, and assumes that the kernel keeps both
busy at once. It does not simulate the chip, and it says nothing about the order of operations,
latency, or contention between kernels.

## 1. Count the work

Count the floating-point operations the kernel performs, $F$. A multiply and an add count as two
FLOPs, so a fused multiply-add counts as two.

## 2. Count the traffic

Count the bytes the kernel moves between main memory and the chip, $Q$. The usual first count
assumes that the kernel reads each input value once and writes each output value once. That
count is usually a lower bound on real traffic, because caches can evict data that the kernel
needs again and stores can trigger extra reads. Real traffic is lower only when an input is still
in cache from an earlier kernel.

## 3. Take the lower ceiling

The arithmetic intensity is $I = F/Q$. With peak compute $P$ and peak bandwidth $\beta$, the
bound is

$$
\text{attainable FLOP/s} = \min(P,\; I \times \beta).
$$

The kernel is memory-bound when $I < P/\beta$ and compute-bound when $I > P/\beta$. The value
$P/\beta$ is the ridge point.

## Limits

The basic model uses one compute peak and one memory level. Real chips have several of each.
A GPU has separate peaks for tensor-core and general-purpose arithmetic, and a CPU has
separate peaks with and without vector instructions. Hierarchical rooflines add one slanted
line per memory level, such as L2 cache and main memory. Very short kernels can also fall far
below the roof, because launch overhead and latency dominate their time.

This course measures its CPU roofline on one core with vectors of 192 MB, so that the data comes
from main memory and every kernel uses the same hardware. The last Try it box in
{doc}`../lectures/01-the-roofline-model` lifts the one-core limit and measures both ceilings on
every core.

:::{warning}
**An intensity counted from the code can be wrong.** The bytes a program requests and the bytes
that actually cross the memory bus differ when caches miss or when stores read their cache line
first. When a kernel lands far from its predicted point, measure its real traffic with a
hardware profiler before drawing conclusions.
:::

## Accuracy

The model gives a bound. A kernel cannot run faster than the roof drawn from the hardware's
true peaks, so the model errs in one direction only. Its authors present it as a guide to which
optimizations can help on a given machine
([Williams et al., CACM '09](https://doi.org/10.1145/1498765.1498785)). In lecture 1, two
memory-bound kernels on one CPU core usually ran at 0.7 to 1.1 times the rate the measured roof
predicted, and in a few runs the vector add fell to about 0.5. A ratio a little above 1 does not break the bound. Each measured ceiling is a lower
bound on the true peak, so another kernel can edge past it. Treat a ratio near 1 as "at the roof" and a ratio well below 1 as
a sign of a bottleneck the model does not include. The trends the model shows are more
reliable than any single absolute number.
