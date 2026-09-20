//! How the game looks to something that learns to play it: a masked
//! [`Observation`], a fixed-width [`ActionSpace`], and an [`Encoder`] that
//! flattens an observation into `f32`s. Kept separate from `core`, which
//! owns the rules and knows nothing about strategy.

pub mod action_index;
pub mod features;
pub mod observation;

pub use action_index::ActionSpace;
pub use features::Encoder;
pub use observation::{rank_index, CellView, GridView, Observation, TurnView, NUM_RANKS};

#[cfg(test)]
mod tests;
