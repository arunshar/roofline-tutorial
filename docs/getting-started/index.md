# How This Course Works

By [Arun Sharma](https://arunshar.com) and [Claude Code](https://claude.com/claude-code)

## The tool

The tool is the roofline model, an analytic bound on how fast a kernel can run. It takes two
counts about the kernel (its FLOPs and the bytes it moves) and two numbers about the chip (its
peak compute and its memory bandwidth). The labs compute the bound with NumPy and also time real
kernels on your CPU to test it.

The model gives the fastest rate a kernel could reach on the hardware's true peaks, so a
measured rate should be at or below that roof. A roof that you measure yourself sits a little
below the true one, so a kernel can land slightly above it. The model does not predict an exact
run time, and the labs run no GPU code. The details are in {doc}`../reference/how-it-works`.

## The helper module

Each lab imports one small module,
[`labs/tutorial_helpers.py`](https://github.com/arunshar/roofline-tutorial/blob/main/labs/tutorial_helpers.py).
It draws the quiz at the end of every lecture:

```python
import tutorial_helpers as th
th.quiz([{"q": "Question?", "options": ["A", "B"], "answer": 0, "explain": "Why A."}])
```

The code for the topic itself stays in the notebooks, where you can read every formula.
{doc}`../reference/api` lists the main functions the labs define and call.

## Where to run

| Option | What you need | First start | Best for |
|---|---|---|---|
| {doc}`Google Colab <run-in-browser>` | a Google account | under a minute | most readers |
| {doc}`Binder <run-in-browser>` | nothing | a few minutes while the server starts | readers without an account |
| {doc}`Your own machine <run-locally>` | Python 3.10 or newer, git | about two minutes | long projects and saving your work |

Each lab uses less than **1 GB of RAM**, and **one CPU core** is enough. All three options
provide both.
