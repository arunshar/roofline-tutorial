# Python API

By [Arun Sharma](https://arunshar.com) and [Claude Code](https://claude.com/claude-code)

The roofline functions are defined in lecture 1, {doc}`../lectures/01-the-roofline-model`, so
you can read every formula next to its use. The quiz comes from
[`labs/tutorial_helpers.py`](https://github.com/arunshar/roofline-tutorial/blob/main/labs/tutorial_helpers.py).
A system is a dict with the keys `name`, `peak_flops` (FLOP/s) and `bandwidth` (bytes/s).
`SYSTEM` also stores `bytes_per_value`, the size of one value in bytes.

## `attainable(intensity, system)`

It returns the roofline bound, in FLOP/s, for one intensity or an array of intensities.

| Argument | Default | Meaning |
|---|---|---|
| `intensity` | required | arithmetic intensity, in FLOP/byte |
| `system` | required | a system dict such as `SYSTEM` or `CPU` |

```python
attainable(10, SYSTEM) / 1e12     # The A100 bound at 10 FLOP/byte is 20.39 TFLOP/s.
```

## `ridge_point(system)`

It returns the intensity, in FLOP/byte, at which the bandwidth and compute limits meet.

| Argument | Default | Meaning |
|---|---|---|
| `system` | required | a system dict |

```python
ridge_point(SYSTEM)               # The A100 ridge point is about 153 FLOP/byte.
```

## `draw_roofline(system, ax=None, label=None)`

It draws the roofline of `system` on log-log axes, in TFLOP/s, and returns the axes.

| Argument | Default | Meaning |
|---|---|---|
| `system` | required | a system dict |
| `ax` | `None` | the matplotlib axes to draw on, or the current axes when `None` |
| `label` | `None` | the legend label, or `system["name"]` when `None` |

```python
ax = draw_roofline(SYSTEM)
draw_roofline(CPU, ax=ax)         # This draws a second roofline on the same axes.
```

## `kernel_costs(v=SYSTEM["bytes_per_value"])`

It returns a dict that maps each of the four example kernels to its FLOPs and the bytes it moves.

| Argument | Default | Meaning |
|---|---|---|
| `v` | `2` | bytes per value |

```python
flops, nbytes = kernel_costs()["vector add"]
flops / nbytes                    # The BF16 vector add does about 0.17 FLOP/byte.
```

## `batched_matvec(batch, m=8192, k=8192, v=SYSTEM["bytes_per_value"])`

It returns the FLOPs and the bytes moved when one $m \times k$ matrix multiplies a batch of
vectors.

| Argument | Default | Meaning |
|---|---|---|
| `batch` | required | number of vectors in the batch |
| `m`, `k` | `8192` | rows and columns of the matrix |
| `v` | `2` | bytes per value |

```python
flops, nbytes = batched_matvec(128)
flops / nbytes                    # A batch of 128 gives about 124 FLOP/byte.
```

## `one_core()`

It returns a context manager that limits NumPy's math library to one thread inside a `with`
block. It calls `threadpool_limits(1)` from
[threadpoolctl](https://github.com/joblib/threadpoolctl) and makes a fresh limiter on each call,
because a limiter stops working after its first `with` block ends.

```python
with one_core():
    t = best_time(lambda: W @ x)  # The matrix-vector product runs on one thread.
```

## `best_time(fn, repeats=5)`

It runs `fn` once to warm up, then returns the fastest of `repeats` timed runs, in seconds.

| Argument | Default | Meaning |
|---|---|---|
| `fn` | required | a function with no arguments, usually a `lambda` |
| `repeats` | `5` | number of timed runs |

```python
best_time(lambda: np.dot(a, b))   # This returns the seconds for one dot product.
```

## `quiz(questions)`

It shows clickable multiple-choice questions in a notebook. Each question is a dict with the keys
`q`, `options` and `answer` (the right option's position, counting from 0), and an optional
`explain`. Every string may hold HTML.

```python
th.quiz([{"q": "Question?", "options": ["A", "B"], "answer": 0, "explain": "Why A."}])
```

The `{quiz}` directive on a Markdown page takes the same list, written in YAML.
