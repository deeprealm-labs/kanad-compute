//! Reconnect backoff: 1 s → 30 s exponential with ±20 % jitter.
//!
//! Matches the Python `ComputeWSClient` reconnect schedule so a Rust
//! gateway behaves like its Python predecessor under flaky networks.

use rand::Rng;
use std::time::Duration;

const INITIAL_MS: u64 = 1_000;
const MAX_MS: u64 = 30_000;
const JITTER_PCT: f64 = 0.20;

#[derive(Debug, Clone)]
pub struct Backoff {
    current_ms: u64,
}

impl Default for Backoff {
    fn default() -> Self {
        Self::new()
    }
}

impl Backoff {
    pub fn new() -> Self {
        Self {
            current_ms: INITIAL_MS,
        }
    }

    /// Compute the next delay (with jitter) and double the base for the
    /// next call, capped at 30 s.
    pub fn next_delay(&mut self) -> Duration {
        let base = self.current_ms as f64;
        let jitter = rand::thread_rng().gen_range(-JITTER_PCT..JITTER_PCT);
        let ms = (base * (1.0 + jitter)).max(0.0) as u64;
        self.current_ms = (self.current_ms.saturating_mul(2)).min(MAX_MS);
        Duration::from_millis(ms)
    }

    /// Call after a successful connect to drop back to the initial delay.
    pub fn reset(&mut self) {
        self.current_ms = INITIAL_MS;
    }

    /// Inspect, mostly for tests.
    pub fn current_base_ms(&self) -> u64 {
        self.current_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramps_exponentially_then_caps() {
        let mut b = Backoff::new();
        let mut last = 0;
        // 1s, 2s, 4s, 8s, 16s, 30s, 30s, ...
        let expected_bases = [1_000, 2_000, 4_000, 8_000, 16_000, 30_000, 30_000];
        for &exp in &expected_bases {
            let prev_base = b.current_base_ms();
            let _ = b.next_delay();
            // Sanity: prev_base matches what we expect *before* doubling.
            assert!(prev_base <= exp + 1);
            last = b.current_base_ms();
        }
        assert_eq!(last, 30_000);
    }

    #[test]
    fn reset_returns_to_initial() {
        let mut b = Backoff::new();
        for _ in 0..3 {
            let _ = b.next_delay();
        }
        b.reset();
        assert_eq!(b.current_base_ms(), 1_000);
    }

    #[test]
    fn jitter_within_bounds() {
        let mut b = Backoff::new();
        for _ in 0..100 {
            let base_before = b.current_base_ms() as f64;
            let d = b.next_delay().as_millis() as f64;
            let low = base_before * (1.0 - JITTER_PCT) - 1.0;
            let high = base_before * (1.0 + JITTER_PCT) + 1.0;
            assert!(d >= low && d <= high, "delay {d} not in [{low},{high}]");
        }
    }
}
