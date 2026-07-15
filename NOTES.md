# Voice Cloning Notes

1. Our fitness function evaluates speaker similarity across multiple sentences and adds an L2 regularization penalty based on drift from the stock voice manifold.
2. This penalty prevents the search from drifting into degraded, raspy, or noisy audio that the embedding encoder might highly score but a human or ASR rejects.
3. We optimize the style tensor in two stages: first by finding the best convex combination (simplex search) of the top-5 stock voices, and then by searching a 256-dimensional shared perturbation.
4. Our best candidate achieved a speaker similarity score of ~0.84-0.86, beating the stock voice baseline decisively while preserving full intelligibility.
5. The similarity plateaued because a style tensor only modulates high-level acoustic features in the frozen generator.
6. Since Kokoro's internal phoneme-to-acoustic network weights are frozen, it cannot learn the specific pronunciations, accents, or fine prosody of the target speaker.
7. Restricting the search to a 256-D shared perturbation (broadcasted to all 510 rows) is crucial; otherwise, independent row perturbations introduce out-of-distribution noise on unseen sentences, triggering ASR failure.
