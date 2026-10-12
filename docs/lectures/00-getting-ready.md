# 0. Getting Ready

By [Arun Sharma](https://arunshar.com) and [Claude Code](https://claude.com/claude-code)

The lecture assumes basic Python with NumPy and a rough idea of how a processor reads data
from memory. A working knowledge of each area is enough. The vocabulary below gives
everyone the same terms, and the sections after it link to readings for any area that feels new.

## Vocabulary

### Work

FLOP
: A FLOP is one floating-point operation, such as one addition or one multiplication. Adding two
  vectors of 1,000 numbers takes 1,000 FLOPs.

FLOP/s
: FLOP/s counts the FLOPs a chip completes per second. One A100 peaks at 312 × 10¹² FLOP/s, or
  312 TFLOP/s, for 16-bit matrix arithmetic.

Kernel
: A kernel is one self-contained computation that runs on the chip, such as a vector add over
  10⁸ numbers or a 4,096 × 4,096 matrix multiply. The roofline model looks at one kernel at a
  time.

### Data

Byte and precision
: A byte is 8 bits. A 32-bit number (FP32) takes 4 bytes, and a 16-bit number (BF16 or FP16)
  takes 2 bytes, so the same vector moves half as many bytes in 16-bit precision.

Main memory
: Main memory is the large memory that holds a program's data outside the chip. It is usually
  HBM on a data-center GPU and GDDR on a gaming GPU, and a CPU's is usually DDR. The A100-80GB
  has 80 GB of HBM2e.

Cache
: A cache is a small, fast memory on the chip that keeps recently used data. An A100 has a
  40 MB L2 cache, about 2,000 times smaller than its 80 GB of HBM2e. Reading a value from cache
  avoids a trip to main memory, which is why caches raise a kernel's effective arithmetic
  intensity.

Memory bandwidth
: Memory bandwidth is the number of bytes per second that main memory can deliver to the chip.
  The A100-80GB SXM's HBM2e delivers about 2,039 GB/s.

## The roofline model

The lecture explains the model from scratch. These readings give the original paper and two
longer treatments for machine learning hardware.

- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785) (Williams, Waterman and Patterson): This 2009 paper introduced the model and tested it on four multicore systems.
- [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/) (Jacob Austin et al.): This chapter of *How To Scale Your Model* applies the model to accelerators and works through examples for matrix multiplies.
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) (Horace He): This essay splits the time that deep learning code loses into compute, memory bandwidth and overhead.

## Hardware numbers

Every roofline needs two numbers about the chip. These pages give the published peaks for the
GPUs in the lecture, and the table lists the values the lecture uses.

- [NVIDIA A100 data sheet](https://www.nvidia.com/en-us/data-center/a100/) (NVIDIA): The data sheet lists peak compute for each precision, memory size and bandwidth.
- [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) (NVIDIA): This page gives the same figures for the newer GPU used in one exercise.

Both pages also list larger tensor-core figures marked "with sparsity". Those figures are twice
the dense peaks, and the lecture uses the dense peaks.

| | A100-80GB SXM | H100 SXM |
|---|---|---|
| Peak dense BF16 compute | 312 TFLOP/s | ~989 TFLOP/s |
| Memory bandwidth | 2,039 GB/s | 3,350 GB/s |
| Ridge point (FLOP/byte) | ~153 | ~295 |

## Are you ready?

These questions check the background that lecture 1 assumes. Pick an answer to see whether it
is right. After a wrong answer, you can try another option.

```{quiz}
- q: How many bytes does a vector of 1,000 BF16 numbers take?
  options: ["1,000 bytes", "2,000 bytes", "4,000 bytes"]
  answer: 1
  explain: Each BF16 number takes 2 bytes, so 1,000 numbers take 2,000 bytes.
- q: How many FLOPs does it take to compute c = a + b for vectors of 1,000 numbers?
  options: ["1,000", "2,000", "3,000"]
  answer: 0
  explain: Each output element needs one addition, so the vector add does 1,000 additions in total.
- q: A chip can do 100 TFLOP/s and a kernel needs 10<sup>12</sup> FLOPs. What is the shortest time the arithmetic can take?
  options: ["10 ms", "100 ms", "1 s"]
  answer: 0
  explain: 10<sup>12</sup> FLOPs ÷ 10<sup>14</sup> FLOP/s = 0.01 s, which is 10 ms.
- q: Memory delivers 2 TB/s and a kernel must read 4 GB. What is the shortest time the reads can take?
  options: ["0.5 ms", "2 ms", "8 ms"]
  answer: 1
  explain: 4 × 10<sup>9</sup> bytes ÷ (2 × 10<sup>12</sup> bytes/s) = 0.002 s, which is 2 ms.
- q: Why can a cache make a kernel faster even when the arithmetic does not change?
  options: ["It raises the chip's peak FLOP/s", "It serves repeated reads without a trip to main memory", "It makes each number smaller"]
  answer: 1
  explain: Data served from the cache does not use main-memory bandwidth, so the kernel moves fewer bytes per FLOP.
```

## Setting up

{doc}`../getting-started/run-in-browser` explains how to open a lab in Colab or Binder, and
{doc}`../getting-started/run-locally` covers a local install. The first cell of lecture 1
checks that your environment works.
