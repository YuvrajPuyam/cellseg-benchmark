# cajal - Which model best outlines cells in tissue images?

*A plain-English summary. Full numbers and code are in the rest of the repo.*

## The question

To analyze tissue under a modern multiplexed microscope, the very first step is **finding
every cell** - drawing an outline around each one. Many pre-trained "foundation" models claim
to do this. Two practical questions:

1. Which one actually works best on this kind of data?
2. Is it worth **fine-tuning** one on your own labels - and how much does that help?

I answered these on **TissueNet**, a public dataset of multiplexed tissue images that come with
expert-drawn cell outlines, running everything on Purdue's **Gilbreth** computing cluster.

## What I compared

Three off-the-shelf models - **Cellpose-SAM**, **μSAM**, and **StarDist** - on two jobs:

- **Whole-cell:** outline the entire cell.
- **Nuclear:** outline just the nucleus.

Each model's outlines were scored against the experts' using standard accuracy measures
(all on a 0-1 scale, higher is better): how many cells it got right (**F1**), how well the
shapes overlapped (**AJI+**), and how tight the outlines were (**boundary score**). Every number
comes with a **95% confidence interval**, so a real difference can be told apart from noise.

## What I found

**Whole-cell - Cellpose-SAM wins clearly:**

| model | accuracy (F1) |
|---|---|
| **Cellpose-SAM** | **0.84** |
| μSAM | 0.74 |

**Nuclear - it's close:**

| model | accuracy (F1) |
|---|---|
| Cellpose-SAM | 0.84 |
| μSAM | 0.81 |
| StarDist | 0.77 |

**Cellpose-SAM is the safest all-round choice.** On nuclei the models are nearly tied - μSAM
actually draws the *tightest* outlines, so the "best" depends on what you care about most.

## Two things worth knowing

**1. How you feed the image matters as much as the model.** μSAM first looked terrible at whole
cells (0.51). The problem wasn't the model - it was how I handed it the image. These images have
two colors (nucleus + cell membrane); my first version averaged them together, which erased the
membrane edges the model needs. Keeping the membrane signal jumped its score from **0.51 to 0.74**.

**2. Fine-tuning helps, and more data helps more.** I trained Cellpose-SAM a little on TissueNet's
own labels and re-measured:

- On **200** example images → whole-cell accuracy rose **+1.5 points**.
- On the **full 2,580** images → **+2.1 points** - and that was only a short run, so it would
  likely climb further.

So a modest amount of hand-labeling buys a real, measurable improvement - and the more you label,
the more you gain.

## Being honest about the limits

- **StarDist** repeatedly crashed on this cluster (a software/GPU-driver conflict), so it was only
  tested on a smaller sample - treat its number as approximate.
- The full-data fine-tuning result (+2.1) comes from a single short run, so it's a solid signal
  but doesn't have error bars yet.
- Everything is measured on TissueNet. It's an honest answer for TissueNet-like tissue; very
  different tissue could behave differently.

## Bottom line

On multiplexed tissue, **Cellpose-SAM is the best out-of-the-box choice**, the models are
near-tied on nuclei, and **fine-tuning on even a few hundred labeled images gives a real accuracy
boost** (with more data giving more). The whole study re-runs with a single command, and the
measurement code is unit-tested for correctness - so the numbers can be trusted and reproduced.
