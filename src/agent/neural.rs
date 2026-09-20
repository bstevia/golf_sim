//! A player driven by a learned value network: 1-ply expectimax using
//! `ValueNet` to score candidate resulting positions, instead of the
//! hand-coded expected-value math `HeuristicAgent` uses.
//!
//! The reveal phase still falls back to `heuristic::choose_reveal` - the
//! network is only consulted for the draw and swap-or-discard decisions.
//!
//! Every candidate position for a decision is evaluated in one batched
//! forward pass rather than one call each - candle's per-call overhead on
//! tiny tensors dominates otherwise, since a single decision can have on the
//! order of a hundred candidates (13 possible stock ranks, each tried
//! against every cell).

use super::heuristic::choose_reveal;
use super::Agent;
use crate::core::card::Rank;
use crate::nn::ValueNet;
use crate::repr::{rank_index, ActionSpace, CellView, Encoder, GridView, Observation, TurnView, NUM_RANKS};
use candle_core::Tensor;
use rand::seq::IndexedRandom;
use rand::Rng;
use rand::RngCore;

/// See `heuristic::PATIENCE_TURNS` for why this exists and why it's keyed
/// off `turns` rather than `reshuffles`.
const PATIENCE_TURNS: u64 = 300;

pub struct NeuralAgent<'a> {
    net: &'a ValueNet,
    encoder: Encoder,
    /// Chance of picking a uniformly random legal action instead of the
    /// network's choice. Zero for play/eval; nonzero during self-play data
    /// generation so training sees more of the state space than the
    /// network's own current opinion would visit.
    pub exploration: f32,
}

impl<'a> NeuralAgent<'a> {
    pub fn new(net: &'a ValueNet, encoder: Encoder) -> Self {
        NeuralAgent { net, encoder, exploration: 0.0 }
    }

    pub fn with_exploration(net: &'a ValueNet, encoder: Encoder, exploration: f32) -> Self {
        NeuralAgent { net, encoder, exploration }
    }

    /// The position after this turn resolves: `cell` set to `rank` if given,
    /// otherwise unchanged (discarding without swapping), in the value
    /// framing `ValueNet` expects (see `nn::observation_for_value`).
    fn snapshot(&self, obs: &Observation, cell: Option<(usize, Rank)>) -> Observation {
        let mut next = crate::nn::observation_for_value(obs);
        if let Some((index, rank)) = cell {
            let grid = &next.grids[0];
            let mut cells = grid.cells().to_vec();
            cells[index] = CellView::Up(rank);
            next.grids[0] = GridView::from_cells(grid.rows(), grid.cols(), &cells);
        }
        next
    }

    /// Evaluates every candidate in one forward pass.
    fn predict_batch(&self, observations: &[Observation]) -> Vec<f32> {
        let dim = self.encoder.len();
        let mut buffer = vec![0.0f32; observations.len() * dim];
        for (i, obs) in observations.iter().enumerate() {
            self.encoder.encode_into(obs, &mut buffer[i * dim..(i + 1) * dim]);
        }
        let input = Tensor::from_slice(&buffer, (observations.len(), dim), self.net.device())
            .expect("building the batch input tensor should not fail");
        let output = self.net.forward(&input).expect("forward pass on well-formed features should not fail");
        output
            .to_vec2::<f32>()
            .expect("value net output should be [batch, 1]")
            .into_iter()
            .map(|row| row[0])
            .collect()
    }
}

impl Agent for NeuralAgent<'_> {
    fn act(&mut self, obs: &Observation, space: &ActionSpace, rng: &mut dyn RngCore) -> usize {
        if self.exploration > 0.0 && rng.random::<f32>() < self.exploration {
            let mask = space.mask(obs);
            let legal: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
            return *legal.choose(rng).expect("every live phase offers at least one legal action");
        }

        match obs.turn {
            TurnView::Reveal { .. } => choose_reveal(obs.own_grid()),

            TurnView::Draw => {
                let grid = obs.own_grid();
                let n = grid.len();

                // One candidate for "stand pat", plus one per (rank, cell)
                // pair. discard_top is always some rank already in this
                // sweep, so it needs no separate pass.
                let mut candidates = Vec::with_capacity(1 + n * NUM_RANKS);
                candidates.push(self.snapshot(obs, None));
                for &r in Rank::ALL.iter() {
                    for i in 0..n {
                        candidates.push(self.snapshot(obs, Some((i, r))));
                    }
                }
                let values = self.predict_batch(&candidates);
                let baseline = values[0];

                let best_after_drawing = |rank: Rank| -> f32 {
                    let start = 1 + rank_index(rank) * n;
                    values[start..start + n].iter().copied().fold(baseline, f32::min)
                };

                let total = obs.unseen_total();
                let stock_value = if total == 0 {
                    baseline
                } else {
                    Rank::ALL
                        .iter()
                        .map(|&r| {
                            let weight = obs.unseen[rank_index(r)] as f32 / total as f32;
                            weight * best_after_drawing(r)
                        })
                        .sum()
                };

                match obs.discard_top {
                    Some(top) if best_after_drawing(top) < stock_value => space.draw_discard(),
                    _ => space.draw_stock(),
                }
            }

            TurnView::Decide { drawn, .. } => {
                let grid = obs.own_grid();
                let n = grid.len();

                let mut candidates = Vec::with_capacity(1 + n);
                candidates.push(self.snapshot(obs, None));
                for i in 0..n {
                    candidates.push(self.snapshot(obs, Some((i, drawn))));
                }
                let values = self.predict_batch(&candidates);

                let mut best_index = space.discard_drawn();
                let mut best_value = values[0];
                for i in 0..n {
                    if values[i + 1] < best_value {
                        best_value = values[i + 1];
                        best_index = i;
                    }
                }

                // Same deadlock risk `HeuristicAgent` had, with the same
                // fix: an untrained (or just unlucky) network has no
                // guaranteed bias toward ever preferring a swap, so once
                // patience runs out, force progress on a face-down cell.
                if best_index == space.discard_drawn() && obs.turns >= PATIENCE_TURNS {
                    let forced = (0..n)
                        .filter(|&i| !grid.cell(i).is_known())
                        .map(|i| (i, values[i + 1]))
                        .min_by(|a, b| a.1.total_cmp(&b.1));
                    if let Some((forced_index, _)) = forced {
                        return forced_index;
                    }
                }
                best_index
            }
        }
    }

    fn name(&self) -> &str {
        "neural"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::GameConfig;
    use crate::core::state::GameState;
    use candle_core::Device;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn neural_agent_always_legal_across_configs() {
        let device = Device::Cpu;
        for config in [
            GameConfig::six_card_golf(2),
            GameConfig { grid_rows: 1, grid_cols: 4, reveal_count: 1, ..GameConfig::six_card_golf(2) },
        ] {
            let encoder = Encoder::new(&config);
            let net = ValueNet::new(encoder.len(), &device).unwrap();
            let space = ActionSpace::new(&config);
            let mut agent = NeuralAgent::new(&net, encoder);

            for seed in 0..10u64 {
                let mut deal_rng = StdRng::seed_from_u64(seed);
                let mut decision_rng = StdRng::seed_from_u64(seed ^ 0xF00D);
                let mut state = GameState::new_with_rng(config.clone(), &mut deal_rng).unwrap();

                let mut steps = 0u64;
                while let Some(obs) = Observation::for_actor(&state) {
                    let index = agent.act(&obs, &space, &mut decision_rng);
                    let legal = state.legal_actions();
                    let action = space.action_at(index, &obs.turn).unwrap_or_else(|| {
                        panic!("seed {seed}: index {index} has no action in {:?}", obs.turn)
                    });
                    assert!(legal.contains(&action), "seed {seed}: chose illegal {action:?}");
                    state.apply(action).unwrap();
                    steps += 1;
                    assert!(steps < 20_000, "seed {seed}: did not terminate");
                }
            }
        }
    }

    /// Regression test for a deadlock class specific to this agent: an
    /// untrained network has no domain-motivated bias like the heuristic's,
    /// so a bad random initialization could get *permanently* stuck always
    /// preferring "draw discard, then discard the same card back" - a loop
    /// that changes neither `stock_remaining` nor `reshuffles`, so a
    /// reshuffle-keyed patience mechanism (the heuristic's original fix)
    /// would never even engage. Fixed by keying patience off `turns`
    /// instead, which advances every turn regardless of draw source. Since
    /// candle's own weight init isn't under our seed control, this sweeps
    /// several fresh random networks rather than relying on one.
    #[test]
    fn neural_agent_terminates_across_several_random_initializations() {
        let device = Device::Cpu;
        let config = GameConfig { grid_rows: 1, grid_cols: 4, reveal_count: 1, ..GameConfig::six_card_golf(2) };

        for trial in 0..8u64 {
            let encoder = Encoder::new(&config);
            let net = ValueNet::new(encoder.len(), &device).unwrap();
            let space = ActionSpace::new(&config);
            let mut agent = NeuralAgent::new(&net, encoder);

            for seed in 0..5u64 {
                let mut deal_rng = StdRng::seed_from_u64(seed);
                let mut decision_rng = StdRng::seed_from_u64(seed ^ 0xF00D);
                let mut state = GameState::new_with_rng(config.clone(), &mut deal_rng).unwrap();

                let mut steps = 0u64;
                while let Some(obs) = Observation::for_actor(&state) {
                    let index = agent.act(&obs, &space, &mut decision_rng);
                    let action = space.action_at(index, &obs.turn).unwrap();
                    state.apply(action).unwrap();
                    steps += 1;
                    assert!(steps < 20_000, "trial {trial} seed {seed}: did not terminate");
                }
            }
        }
    }

    #[test]
    fn full_exploration_still_only_takes_legal_actions() {
        let device = Device::Cpu;
        let config = GameConfig::six_card_golf(2);
        let encoder = Encoder::new(&config);
        let net = ValueNet::new(encoder.len(), &device).unwrap();
        let space = ActionSpace::new(&config);
        let mut agent = NeuralAgent::with_exploration(&net, encoder, 1.0);

        let mut deal_rng = StdRng::seed_from_u64(1);
        let mut decision_rng = StdRng::seed_from_u64(2);
        let mut state = GameState::new_with_rng(config, &mut deal_rng).unwrap();
        while let Some(obs) = Observation::for_actor(&state) {
            let index = agent.act(&obs, &space, &mut decision_rng);
            let action = space.action_at(index, &obs.turn).unwrap();
            assert!(state.legal_actions().contains(&action));
            state.apply(action).unwrap();
        }
    }
}
