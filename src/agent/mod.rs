//! Things that play Golf, and a way to measure how well they do it.
//! Everything here sees only an [`Observation`], never a [`GameState`].

pub mod eval;
pub mod heuristic;
pub mod neural;
pub mod random;

pub use eval::{eval_head_to_head, play_match, MatchResult};
pub use heuristic::HeuristicAgent;
pub use neural::NeuralAgent;
pub use random::RandomAgent;

use crate::repr::{ActionSpace, Observation};
use rand::RngCore;

/// Anything that can play Golf. `act` returns a flat index into `space`, in
/// `obs`'s own cell numbering - if you canonicalize `obs` internally, map
/// your choice back through the permutation before returning it.
///
/// Uses `&mut dyn RngCore` instead of a generic so `Agent` is object-safe.
pub trait Agent {
    fn act(&mut self, obs: &Observation, space: &ActionSpace, rng: &mut dyn RngCore) -> usize;

    /// A short, stable name for reporting eval results.
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests;
