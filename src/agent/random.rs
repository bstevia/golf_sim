use super::Agent;
use crate::repr::{ActionSpace, Observation};
use rand::seq::IndexedRandom;
use rand::RngCore;

#[derive(Debug, Default, Clone, Copy)]
pub struct RandomAgent;

impl Agent for RandomAgent {
    fn act(&mut self, obs: &Observation, space: &ActionSpace, rng: &mut dyn RngCore) -> usize {
        let mask = space.mask(obs);
        let legal: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
        *legal
            .choose(rng)
            .expect("every live phase offers at least one legal action")
    }

    fn name(&self) -> &str {
        "random"
    }
}
