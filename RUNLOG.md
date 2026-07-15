# Run Log

### Run 1: Stock Voice Baseline & Naive Blend (blend.py)
* **Fitness Design**: Cosine similarity to target embedding on a single sentence.
* **Settings**: Baseline code (`blend.py`). Naive 50/50 blend of top-2 voices.
* **Score**: 0.7180 (Best stock voice: 0.7010).
* **What I heard**: Clear, natural, high-quality audio, but the voice characteristics only weakly resembled the target speaker.
* **What I changed**: Concluded that we need to search the weights of more than 2 voices and add a regularizer.

### Run 2: Phase 1 Simplex Weight Blend Optimization
* **Fitness Design**: Multi-sentence average speaker similarity (3 sentences).
* **Settings**: 40 iterations on a 5- simplex weights of the top-5 ranked stock voices.
* **Score**: 0.7950.
* **What I heard**: Clean and natural audio. The blend sounds like a hybrid speaker that exhibits significantly stronger vocal resemblance to the target speaker than any single stock voice.
* **What I changed**: Used this optimized blend as the starting point for fine-grained perturbation search.

### Run 3: Shared 256-D Perturbation Walk (No Regularization)
* **Fitness Design**: Multi-sentence speaker similarity only.
* **Settings**: 110 iterations, step=0.03, perturbing all rows independently (full 510x256 space).
* **Score**: 0.8910 (on training sentences), but degraded on unseen sentences.
* **What I heard**: The audio became increasingly metallic, raspy, and full of clicking artifacts. Unseen sentences were completely distorted, which would fail the ASR intelligibility gate (WER > 25%).
* **What I changed**: Switched to a shared 256-D perturbation vector (broadcast to all 510 rows) and added an L2 manifold regularization penalty.

### Run 4: Shared 256-D Perturbation Walk (With Regularization - Final)
* **Fitness Design**: Average speaker similarity on 3 sentences minus L2 norm drift penalty (weight=0.15, max_drift=0.20) from the optimized stock blend.
* **Settings**: 110 iterations, step=0.02 (annealing down to 0.01), shared 256-D perturbation.
* **Score**: 0.8520.
* **What I heard**: Very clean, clear, and highly intelligible speech (ASR WER < 5%). The voice similarity remains high and mimics the target speaker's vocal tone and weight remarkably well without any raspiness.
