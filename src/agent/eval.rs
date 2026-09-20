use super::Agent;
use crate::core::config::GameConfig;
use crate::core::state::GameState;
use crate::repr::{ActionSpace, Observation};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::RngCore;
use std::fmt;

/// Plays one hand to completion. Panics if an agent returns an action that
/// isn't legal - in an eval harness that's a bug worth failing loudly on.
pub fn play_match<R: RngCore>(
    config: &GameConfig,
    agents: &mut [&mut dyn Agent],
    deal_rng: &mut R,
    decision_rng: &mut dyn RngCore,
) -> Vec<i32> {
    assert_eq!(agents.len(), config.num_players, "need exactly one agent per seat");
    let space = ActionSpace::new(config);
    let mut state = GameState::new_with_rng(config.clone(), deal_rng).expect("validated config");

    while let Some(obs) = Observation::for_actor(&state) {
        let seat = obs.seat;
        let index = agents[seat].act(&obs, &space, decision_rng);
        let action = space
            .action_at(index, &obs.turn)
            .unwrap_or_else(|| panic!("{}: index {index} has no action in {:?}", agents[seat].name(), obs.turn));
        if let Err(err) = state.apply(action) {
            panic!("{}: chose {action:?}, illegal in {:?}: {err}", agents[seat].name(), obs.turn);
        }
    }
    state.scores()
}

/// Head-to-head result for two agents, over `2 * hands` samples (each hand
/// played with both seat assignments - see the module doc).
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub agent_a_name: String,
    pub agent_b_name: String,
    pub hands_dealt: u64,
    pub mean_score_a: f64,
    pub mean_score_b: f64,
    /// Half-width of a 95% confidence interval on the mean.
    pub ci95_a: f64,
    pub ci95_b: f64,
    /// Mean of (A - B), paired hand by hand. Lower is better, so negative
    /// means A is ahead.
    pub mean_diff_a_minus_b: f64,
    pub diff_ci95: f64,
    /// Share of samples each agent strictly won. Ties count toward neither.
    pub win_rate_a: f64,
    pub win_rate_b: f64,
    pub tie_rate: f64,
}

impl fmt::Display for MatchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} vs {} over {} dealt hands (x2 seat assignments):", self.agent_a_name, self.agent_b_name, self.hands_dealt)?;
        writeln!(
            f,
            "  {}: mean {:.3} +/- {:.3}, win rate {:.1}%",
            self.agent_a_name, self.mean_score_a, self.ci95_a, 100.0 * self.win_rate_a
        )?;
        writeln!(
            f,
            "  {}: mean {:.3} +/- {:.3}, win rate {:.1}%",
            self.agent_b_name, self.mean_score_b, self.ci95_b, 100.0 * self.win_rate_b
        )?;
        writeln!(f, "  ties: {:.1}%", 100.0 * self.tie_rate)?;
        write!(
            f,
            "  A - B: {:.3} +/- {:.3} ({})",
            self.mean_diff_a_minus_b,
            self.diff_ci95,
            if self.mean_diff_a_minus_b + self.diff_ci95 < 0.0 {
                "A significantly better at 95%"
            } else if self.mean_diff_a_minus_b - self.diff_ci95 > 0.0 {
                "B significantly better at 95%"
            } else {
                "not significant at 95%"
            }
        )
    }
}

/// Mean and 95% CI half-width via the normal approximation.
fn mean_and_ci95(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    if xs.len() < 2 {
        return (mean, f64::INFINITY);
    }
    let variance = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, 1.96 * (variance / n).sqrt())
}

/// Runs `agent_a` against `agent_b` over `hands` deals, each replayed with
/// both seat assignments under a fixed `seed`. Two players only; use
/// [`play_match`] directly for other player counts.
pub fn eval_head_to_head(
    config: &GameConfig,
    agent_a: &mut dyn Agent,
    agent_b: &mut dyn Agent,
    hands: u64,
    seed: u64,
) -> MatchResult {
    assert_eq!(config.num_players, 2, "eval_head_to_head is for exactly two players");
    assert!(hands >= 1, "need at least one dealt hand");

    let mut decision_rng = StdRng::seed_from_u64(seed ^ 0xD1CE_D1CE_D1CE_D1CE); // arbitrary, just distinct from deal seeds

    let mut scores_a = Vec::with_capacity(2 * hands as usize);
    let mut scores_b = Vec::with_capacity(2 * hands as usize);
    let (mut wins_a, mut wins_b, mut ties) = (0u64, 0u64, 0u64);

    for hand in 0..hands {
        let deal_seed = seed
            .wrapping_add(hand)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);

        let mut deal_rng = StdRng::seed_from_u64(deal_seed);
        let scores = play_match(config, &mut [agent_a, agent_b], &mut deal_rng, &mut decision_rng);
        record(&mut scores_a, &mut scores_b, &mut wins_a, &mut wins_b, &mut ties, scores[0], scores[1]);

        // same deal_seed: identical shoe, seats swapped
        let mut deal_rng = StdRng::seed_from_u64(deal_seed);
        let scores = play_match(config, &mut [agent_b, agent_a], &mut deal_rng, &mut decision_rng);
        record(&mut scores_a, &mut scores_b, &mut wins_a, &mut wins_b, &mut ties, scores[1], scores[0]);
    }

    let (mean_score_a, ci95_a) = mean_and_ci95(&scores_a);
    let (mean_score_b, ci95_b) = mean_and_ci95(&scores_b);
    let diffs: Vec<f64> = scores_a.iter().zip(&scores_b).map(|(a, b)| a - b).collect();
    let (mean_diff_a_minus_b, diff_ci95) = mean_and_ci95(&diffs);
    let samples = scores_a.len() as f64;

    MatchResult {
        agent_a_name: agent_a.name().to_string(),
        agent_b_name: agent_b.name().to_string(),
        hands_dealt: hands,
        mean_score_a,
        mean_score_b,
        ci95_a,
        ci95_b,
        mean_diff_a_minus_b,
        diff_ci95,
        win_rate_a: wins_a as f64 / samples,
        win_rate_b: wins_b as f64 / samples,
        tie_rate: ties as f64 / samples,
    }
}

#[allow(clippy::too_many_arguments)]
fn record(
    scores_a: &mut Vec<f64>,
    scores_b: &mut Vec<f64>,
    wins_a: &mut u64,
    wins_b: &mut u64,
    ties: &mut u64,
    score_a: i32,
    score_b: i32,
) {
    scores_a.push(score_a as f64);
    scores_b.push(score_b as f64);
    match score_a.cmp(&score_b) {
        std::cmp::Ordering::Less => *wins_a += 1,
        std::cmp::Ordering::Greater => *wins_b += 1,
        std::cmp::Ordering::Equal => *ties += 1,
    }
}
