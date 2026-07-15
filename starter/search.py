import argparse
import os
import random
import numpy as np
import torch
import soundfile as sf

import synth
import similarity as sim

# Multi-sentence validation to avoid overfitting
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Please confirm your order number after the beep.",
    "I will call you back tomorrow at three thirty.",
]

def get_simplex_weights(k):
    """Generate random weights that sum to 1.0 (Dirichlet distribution)."""
    w = np.random.exponential(1.0, k)
    return w / np.sum(w)

def fitness(voice, target_emb, texts, baseline_tensor=None, max_drift=0.25):
    """
    Computes average similarity to the target speaker over multiple sentences.
    If baseline_tensor is provided, penalizes candidate tensors that drift too
    far from the baseline stock voice manifold to prevent degraded, raspy audio.
    """
    # 1. Synthesize and evaluate speaker similarity
    scores = []
    for text in texts:
        try:
            wav = synth.synthesize(text, voice)
            # Prevent empty or silent audio
            if len(wav) == 0 or np.std(wav) < 1e-4:
                return -1.0
            s = sim.similarity_to_target(wav, target_emb)
            scores.append(s)
        except Exception as e:
            # Handle numerical instability errors gracefully
            return -1.0
            
    avg_sim = float(np.mean(scores))
    
    # 2. Prevent gaming by adding a soft manifold penalty
    # This keeps our searched tensor close to the natural voices space
    penalty = 0.0
    if baseline_tensor is not None:
        diff = voice - baseline_tensor
        l2_diff = float(torch.norm(diff))
        # Soft margin penalty for L2 drift beyond threshold
        if l2_diff > max_drift:
            penalty = 0.15 * (l2_diff - max_drift)
            
    return avg_sim - penalty

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--out", default="voice.pt")
    ap.add_argument("--listen_every", type=int, default=5)
    args = ap.parse_args()

    # Get target speaker embedding
    target = sim.target_embedding(args.reference_dir)
    
    # Load all stock voices
    voices = synth.stock_voices()
    print(f"Loaded {len(voices)} stock voices.")

    # Step 1: Rank stock voices against target (single sentence baseline check)
    print("Ranking stock voices...")
    rankings = []
    for name, v in voices.items():
        wav = synth.synthesize(SENTENCES[0], v)
        s = sim.similarity_to_target(wav, target)
        rankings.append((s, name))
    rankings.sort(reverse=True)
    
    print("\nTop 5 Stock Voices:")
    for s, name in rankings[:5]:
        print(f"  {name:20s} {s:.4f}")

    top_names = [name for _, name in rankings[:5]]
    top_tensors = [voices[name] for name in top_names]
    
    # Phase 1: Convex Simplex Weight Search (50 iterations)
    # Search for the best blending of the top 5 stock voices
    print("\n=== Phase 1: Optimizing Convex Blend Weights ===")
    best_blend_f = -1.0
    best_weights = None
    
    # Initial baseline is the single best stock voice (weights: 1, 0, 0, 0, 0)
    init_weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    init_blend = top_tensors[0]
    best_blend_f = fitness(init_blend, target, SENTENCES)
    best_weights = init_weights
    print(f"Single best voice fitness: {best_blend_f:.4f}")
    
    # We spend 40 iterations optimizing the blend weights
    blend_iters = min(40, args.iters // 3)
    for i in range(1, blend_iters + 1):
        # Perturb the best weights slightly or sample new ones
        if random.random() < 0.3:
            cand_weights = get_simplex_weights(5)
        else:
            # Small random walk on the simplex
            noise = np.random.normal(0, 0.08, 5)
            cand_weights = np.clip(best_weights + noise, 0, None)
            cand_weights /= np.sum(cand_weights) + 1e-9
            
        cand_blend = sum(w * t for w, t in zip(cand_weights, top_tensors))
        f = fitness(cand_blend, target, SENTENCES)
        
        if f > best_blend_f:
            best_blend_f = f
            best_weights = cand_weights
            print(f"  Iter {i:2d} | New Best Blend Weights: {best_weights} | Fitness: {best_blend_f:.4f}")
            
    # Save the best blend tensor as our starting point
    best_blend_tensor = sum(w * t for w, t in zip(best_weights, top_tensors))
    print(f"Phase 1 complete. Best Blend Fitness: {best_blend_f:.4f}")

    # Phase 2: Shared 256-D Perturbation Walk (Remaining iterations)
    # We apply the SAME 256-D perturbation to all 510 rows. This keeps the voice
    # stable and consistent across sentences with different phoneme counts.
    print("\n=== Phase 2: Shared 256-D Perturbation Walk ===")
    best_voice = best_blend_tensor.clone()
    best_f = best_blend_f
    
    # Start the random walk
    step_size = 0.02
    accepted = 0
    perturb_iters = args.iters - blend_iters
    
    # Shared perturbation vector starts at zero
    best_delta = torch.zeros(1, 256)
    
    for i in range(1, perturb_iters + 1):
        # Step size annealing over time
        current_step = step_size * (1.0 - (i / perturb_iters) * 0.5)
        
        # Perturb the shared 256-dimensional vector
        cand_delta = best_delta + current_step * torch.randn(1, 256)
        
        # Broadcast the addition across all 510 rows
        cand_voice = best_blend_tensor + cand_delta
        
        # Fitness with regularization penalty to prevent drifting into raspy/degraded audio
        f = fitness(cand_voice, target, SENTENCES, baseline_tensor=best_blend_tensor, max_drift=0.20)
        
        if f > best_f:
            best_voice = cand_voice
            best_delta = cand_delta
            best_f = f
            accepted += 1
            print(f"  Iter {i:3d} | Accepted #{accepted:2d} | Fitness: {best_f:.4f} (drift: {float(torch.norm(best_delta)):.3f})")
            
            if accepted % args.listen_every == 0:
                sf.write(f"listen_{accepted}.wav", synth.synthesize(SENTENCES[0], best_voice), synth.SR)
                print(f"    -> saved checkpoint: listen_{accepted}.wav")

    # Save final results
    torch.save(best_voice, args.out)
    sf.write("listen_final.wav", synth.synthesize(SENTENCES[0], best_voice), synth.SR)
    print(f"\nFinal Voice Cloning completed!")
    print(f"Saved optimized style tensor to: {args.out}")
    print(f"Saved demo audio to: listen_final.wav")
    print(f"Final Fitness Score: {best_f:.4f}")

if __name__ == "__main__":
    main()
