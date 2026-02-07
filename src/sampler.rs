// Sampling strategies: temperature, top-k, top-p, with xorshift64 PRNG.

use crate::tensor;

pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

pub struct Sampler {
    config: SamplerConfig,
    rng_state: u64,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        let rng_state = if config.seed == 0 { 42 } else { config.seed };
        Sampler { config, rng_state }
    }

    /// Sample a token index from the logits vector.
    pub fn sample(&mut self, logits: &mut Vec<f32>) -> u32 {
        // Temperature = 0 means greedy (argmax)
        if self.config.temperature <= 0.0 {
            return argmax(logits);
        }

        // Apply temperature
        for v in logits.iter_mut() {
            *v /= self.config.temperature;
        }

        // Top-k filtering
        if self.config.top_k > 0 && self.config.top_k < logits.len() {
            self.apply_top_k(logits);
        }

        // Convert to probabilities
        tensor::softmax(logits);

        // Top-p (nucleus) filtering
        if self.config.top_p < 1.0 {
            self.apply_top_p(logits);
        }

        // Weighted random sampling
        self.sample_from_probs(logits)
    }

    /// Keep only the top-k logits, set the rest to -inf.
    fn apply_top_k(&self, logits: &mut Vec<f32>) {
        let k = self.config.top_k;

        // Find the k-th largest value using partial sort
        let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let threshold = indexed[k - 1].1;
        for v in logits.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    /// Top-p (nucleus) sampling: zero out probabilities outside the nucleus.
    fn apply_top_p(&self, probs: &mut Vec<f32>) {
        // Sort indices by probability descending
        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumsum = 0.0f32;
        let mut cutoff_idx = indexed.len();
        for (i, &(_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum >= self.config.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out everything after the cutoff
        for &(idx, _) in indexed[cutoff_idx..].iter() {
            probs[idx] = 0.0;
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for v in probs.iter_mut() {
                *v /= sum;
            }
        }
    }

    /// Weighted random sampling from a probability distribution.
    fn sample_from_probs(&mut self, probs: &[f32]) -> u32 {
        let r = self.random_f32();
        let mut cumsum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i as u32;
            }
        }
        // Fallback: return the last non-zero probability token
        (probs.len() - 1) as u32
    }

    /// Xorshift64 PRNG: returns a random f32 in [0, 1).
    fn random_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f32) / (u64::MAX as f32)
    }
}

fn argmax(values: &[f32]) -> u32 {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

#[cfg(test)]
mod tests {
    use super::{Sampler, SamplerConfig};

    fn make_logits(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let x = (s as f32) / (u64::MAX as f32);
            out.push(x * 8.0 - 4.0);
        }
        out
    }

    #[test]
    fn temperature_zero_is_always_greedy() {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 0.1,
            seed: 123,
        });
        let mut logits = vec![-1.0f32, 2.0f32, 0.5f32, 1.9f32];
        let token = sampler.sample(&mut logits);
        assert_eq!(token, 1);
    }

    #[test]
    fn fixed_seed_sampling_is_reproducible() {
        let cfg = SamplerConfig {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            seed: 42,
        };
        let mut a = Sampler::new(SamplerConfig {
            temperature: cfg.temperature,
            top_k: cfg.top_k,
            top_p: cfg.top_p,
            seed: cfg.seed,
        });
        let mut b = Sampler::new(SamplerConfig {
            temperature: cfg.temperature,
            top_k: cfg.top_k,
            top_p: cfg.top_p,
            seed: cfg.seed,
        });

        for step in 0..200u64 {
            let mut logits_a = make_logits(step * 7919 + 17, 128);
            let mut logits_b = logits_a.clone();
            let ta = a.sample(&mut logits_a);
            let tb = b.sample(&mut logits_b);
            assert_eq!(ta, tb, "mismatch at step {}", step);
        }
    }

    #[test]
    fn different_seed_changes_sampling_path() {
        let mut a = Sampler::new(SamplerConfig {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            seed: 1,
        });
        let mut b = Sampler::new(SamplerConfig {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            seed: 2,
        });

        let mut differs = false;
        for step in 0..120u64 {
            let mut logits_a = make_logits(step * 104729 + 31, 96);
            let mut logits_b = logits_a.clone();
            let ta = a.sample(&mut logits_a);
            let tb = b.sample(&mut logits_b);
            if ta != tb {
                differs = true;
                break;
            }
        }

        assert!(
            differs,
            "different seeds unexpectedly produced identical path"
        );
    }
}
