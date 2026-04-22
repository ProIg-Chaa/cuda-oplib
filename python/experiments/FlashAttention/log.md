# FlashAttention Experiment Log

## Goal

This log records the discussion, reasoning, and current implementation state for
the `FlashAttention` experiment path under:

- `python/experiments/FlashAttention/flashattention.cu`

The main goal of this round was:

- start from a correct naive attention baseline
- understand what online softmax really maintains
- understand how multiple threads can cooperatively maintain one output row
- prepare for a future transition from naive attention to tiled / flash-style attention

---

## 1. Naive Attention Phase

The first kernel under discussion was a naive float32 attention implementation:

- one block handled one query row
- first materialized all `score[k] = q · k / sqrt(D)`
- then computed softmax in multiple passes
- finally computed:
  - `O[q, d] = sum_k softmax(score[k]) * V[k, d]`

### Main issues found during review

1. There was an initial compile error:
   - loop condition accidentally used `k` instead of `k_row`

2. There was initially a write race on output:
   - multiple threads did `o_ptr[d] += ...`
   - this was later fixed by switching output accumulation to "threads split the `d` dimension"

3. `__syncthreads()` was originally placed inside loops whose iteration counts could differ between threads:
   - this could deadlock the block
   - the synchronization points later had to be moved outside those loops

4. The standard scaling factor was missing at first:
   - `1 / sqrt(D)`
   - this was later added

5. The whole `score[Sk]` vector was stored in shared memory:
   - acceptable only for very small shapes
   - not scalable

### Final judgment on the naive kernel

By the end of the naive-attention review, the kernel was considered:

- basically correct as a small-shape baseline
- useful as a learning reference
- not scalable enough to be considered a serious attention kernel

The key conclusion was:

> This naive kernel is acceptable as a correctness / teaching baseline, but not
> as a long-term direction for performance work.

---

## 2. Why Online Softmax Was Introduced

The next design goal was to avoid materializing the full score vector and move
toward an online softmax / online attention update.

The core idea discussed was:

- do not maintain all softmax probabilities explicitly
- instead maintain:
  - `m`: running max score
  - `l`: running denominator
  - `acc`: running weighted sum of values

The final output row is:

- `O = acc / l`

This is the critical conceptual shift from naive attention to online attention.

---

## 3. Online Softmax State Design

The state eventually became conceptually:

```cpp
struct OnlineSfstate {
    float max;
    float exp_sum;
};
```

Where:

- `max` corresponds to the running max score
- `exp_sum` corresponds to:
  - `sum exp(score_i - max)`

### Important naming correction

Earlier discussion noted that a name like `local_log_sum` would be misleading.
The quantity being maintained here is **not** a log-sum-exp value, but the
already exponentiated normalized sum.

Renaming it to `exp_sum` was the correct improvement.

---

## 4. How Online Updates Work

Two update cases were clarified.

### Case A: new score does not exceed current max

If:

- `score <= state.max`

Then:

- max stays unchanged
- denominator adds one extra term
- accumulator adds one extra weighted value vector

Formula:

- `exp_diff = exp(score - state.max)`
- `state.exp_sum += exp_diff`
- `acc += exp_diff * V`

### Case B: new score becomes new max

If:

- `score > state.max`

Then:

- all old contributions must be rescaled by:
  - `exp(old_max - new_score)`

Formula:

- `exp_diff = exp(state.max - score)`
- `state.exp_sum = state.exp_sum * exp_diff + 1`
- `state.max = score`
- `acc = acc * exp_diff + V`

This logic was confirmed as mathematically correct.

---

## 5. Update vs Combine

An important conceptual confusion was resolved during discussion:

- `update` is **thread-local / state-local**
- `combine` is **state-to-state merge**

### `update`

This means:

- take one existing state
- incorporate one new score / one new value row

Formally:

- `state <- update(state, score_k, V_k)`

### `combine`

This means:

- take two already-accumulated partial states
- merge them into one consistent state

Formally:

- `state <- combine(state_a, state_b)`

This distinction matters because an online attention implementation may or may
not actually need a cross-thread combine depending on how the work is split.

---

## 6. Multiple Threads Maintaining One Output Row

One of the most important implementation discussions was:

> How can one output vector `O[q, :]` be maintained by many threads?

The final clarified principle was:

- softmax weights are shared across the whole query row
- output dimensions `d` are partitioned across threads

So:

- all threads conceptually use the same score / same softmax state
- each thread only maintains the `acc` values for the dimensions it owns

This led to the introduction of:

```cpp
template <int MAX_FRAG>
struct VecFragment {
    float acc[MAX_FRAG];
    int d_idx[MAX_FRAG];
    int valid;
};
```

Meaning:

- one thread owns a small fragment of output dimensions
- `d_idx[i]` tells which global output dimension is mapped to `acc[i]`
- `valid` tells how many entries are actually used

This was considered the right direction for a teaching-friendly online attention
prototype.

---

## 7. Fragment Initialization Logic

The agreed fragment split strategy was:

- thread `tid` owns dimensions:
  - `d = tid, tid + BLOCK_SIZE, tid + 2*BLOCK_SIZE, ...`

This was implemented through `vec_fragment_init(...)`.

### Important limitation

One caveat was identified:

- if `MAX_FRAG` is too small, some dimensions will be silently dropped

This is a real risk in the current code and must remain explicitly remembered.

Current practical constraint:

- `MAX_FRAG` must be large enough so that each thread can cover all the `d`
  indices assigned to it

Otherwise:

- the kernel may "run" but produce incomplete output rows

---

## 8. Fragment-Based Helper Functions

The following helper set was gradually shaped:

- `online_sfstate_init()`
- `vec_fragment_init(...)`
- `online_frafs_update(...)`
- `write_back_output(...)`
- `online_frafs_combine(...)`

### Current interpretation

- `online_sfstate_init()`
  - correct
- `vec_fragment_init(...)`
  - correct under the `MAX_FRAG` capacity assumption
- `online_frafs_update(...)`
  - mathematically correct
- `write_back_output(...)`
  - correct as the final normalization step
- `online_frafs_combine(...)`
  - mathematically correct **under a strong assumption**

### Strong assumption for `online_frafs_combine(...)`

The combine helper only makes sense if:

- `frag_a` and `frag_b` refer to the same logical set of output dimensions
- or at least the overlapping dimensions are deliberately intended to be merged

In the current project discussion, this was accepted as a constrained helper,
not a fully general one.

---

## 9. The Main Structural Mistake in the First Online Kernel Attempts

When the first online attention kernel body was checked, the main error was:

- the same `score[k_row]` was being updated multiple times per thread because
  an extra loop over dimensions was wrapped around `online_frafs_update(...)`

That was wrong because:

- the fragment helper already updates all dimensions owned by that thread
- one `score[k_row]` should correspond to one update call per thread

The corrected logic became:

- for each `k_row`
  - compute / access one score
  - get one `V[k_row, :]`
  - call `online_frafs_update(...)` exactly once per thread

Another mistake in the early online kernel was:

- writing back the output inside the `k_row` loop

This was corrected conceptually to:

- process all keys first
- write back once at the end

---

## 10. Current Interpretation of the Online Kernel

The current online kernel structure is still a **transition version**, not yet
the final desired form.

The present logic still:

- materializes the full `score[k]` vector in shared memory first
- then runs online updates over that stored score vector

So it is not yet "fully online" in the strongest sense.

However, it is still useful because it lets us verify:

- helper correctness
- output fragmentation logic
- thread-to-dimension ownership
- online softmax state updates

### Important nuance

At the moment, each thread still maintains its own `local_state`:

- same sequence of scores
- same update order
- different output fragments

Because all threads currently iterate over the full `k_row` range, their scalar
states are expected to numerically match.

This means the current kernel may still produce numerically correct results.

But structurally, it is still redundant:

- the softmax scalar state is effectively duplicated across threads

This is acceptable as an intermediate learning stage, but not the final design.

---

## 11. Current Main Open Problems

These are the main issues still active after the latest review:

1. The online kernel still duplicates scalar state across threads
   - `max`
   - `exp_sum`

2. The online kernel still materializes all scores before updating
   - not yet the intended streaming form

3. `MAX_FRAG` capacity must be manually kept consistent with thread-to-dimension
   mapping

4. There is still mixed "learning sample" code and "mainline experiment" code
   inside the same file

5. The code should eventually decide on one primary kernel path:
   - fully naive baseline
   - fragment-based online baseline
   - future tiled / flash-style kernel

---

## 12. Current Best Summary

At the current point in development:

- the naive attention baseline is basically correct for small shapes
- the online softmax math is understood and helper formulas are mostly correct
- fragment-based output ownership is the right direction
- the current online kernel is a valid intermediate prototype
- but it is not yet a clean final online attention implementation

The most important practical understanding achieved so far is:

> Softmax scalar state is shared conceptually across the row, while the output
> accumulator is partitioned by output dimensions across threads.

That is the key principle to preserve in future refactors.

---

## 13. Recommended Next Step

The next step should not be "more optimization" yet.

The next step should be:

1. make the online kernel logically clean
2. clearly separate baseline vs online prototype code
3. reduce redundant per-thread scalar-state duplication
4. only after that, start thinking about true tiled / flash-style online updates

---

## 14. Practical Reminder For Future Work

When coming back to this file, check these first:

1. Is the kernel still materializing the whole `score` vector?
2. Is `MAX_FRAG` large enough for the chosen `(BLOCK_SIZE, D)`?
3. Is each `score[k_row]` updated exactly once per thread?
4. Is output written back only once, after all `k_row` are processed?
5. Are the scalar softmax states duplicated per thread or actually shared / merged?

These five questions should be enough to quickly re-enter the problem.
