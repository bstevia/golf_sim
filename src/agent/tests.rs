use super::*;
use crate::core::config::GameConfig;
use crate::core::state::GameState;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Runs an agent through many hands, checking every `act()` call resolves to
/// a legal engine action.
fn assert_agent_only_takes_legal_actions(agent: &mut dyn Agent, config: GameConfig, seeds: std::ops::Range<u64>) {
    let space = ActionSpace::new(&config);
    for seed in seeds {
        let mut deal_rng = StdRng::seed_from_u64(seed);
        let mut decision_rng = StdRng::seed_from_u64(seed ^ 0xABCD);
        let mut state = GameState::new_with_rng(config.clone(), &mut deal_rng).unwrap();

        while let Some(obs) = Observation::for_actor(&state) {
            let index = agent.act(&obs, &space, &mut decision_rng);
            let legal = state.legal_actions();
            let action = space.action_at(index, &obs.turn).unwrap_or_else(|| {
                panic!("seed {seed}: {} returned index {index} with no action in {:?}", agent.name(), obs.turn)
            });
            assert!(legal.contains(&action), "seed {seed}: {} chose illegal {action:?}", agent.name());
            state.apply(action).unwrap();
        }
    }
}

#[test]
fn random_agent_always_legal_across_configs() {
    for config in [
        GameConfig::six_card_golf(2),
        GameConfig { grid_rows: 3, grid_cols: 3, reveal_count: 3, ..GameConfig::six_card_golf(4) },
        GameConfig { grid_rows: 1, grid_cols: 4, reveal_count: 1, ..GameConfig::six_card_golf(2) },
    ] {
        assert_agent_only_takes_legal_actions(&mut RandomAgent, config, 0..15);
    }
}

#[test]
fn heuristic_agent_always_legal_across_configs() {
    for config in [
        GameConfig::six_card_golf(2),
        GameConfig { grid_rows: 3, grid_cols: 3, reveal_count: 3, ..GameConfig::six_card_golf(4) },
        GameConfig { grid_rows: 1, grid_cols: 4, reveal_count: 1, ..GameConfig::six_card_golf(2) },
        GameConfig { reveal_count: 0, ..GameConfig::six_card_golf(3) },
    ] {
        assert_agent_only_takes_legal_actions(&mut HeuristicAgent, config, 0..15);
    }
}

#[test]
fn eval_head_to_head_is_deterministic() {
    let config = GameConfig::six_card_golf(2);
    let run = || {
        let mut a = HeuristicAgent;
        let mut b = RandomAgent;
        eval_head_to_head(&config, &mut a, &mut b, 40, 123)
    };
    let first = run();
    let second = run();
    assert_eq!(first.mean_score_a, second.mean_score_a);
    assert_eq!(first.mean_score_b, second.mean_score_b);
    assert_eq!(first.win_rate_a, second.win_rate_a);
}

/// If this ever fails, something in the heuristic's expected-value logic
/// broke - requires the whole 95% CI below zero, not just the point estimate.
#[test]
fn heuristic_significantly_beats_random() {
    let config = GameConfig::six_card_golf(2);
    let mut heuristic = HeuristicAgent;
    let mut random = RandomAgent;
    let result = eval_head_to_head(&config, &mut heuristic, &mut random, 300, 7);

    assert!(
        result.mean_diff_a_minus_b + result.diff_ci95 < 0.0,
        "heuristic should significantly beat random: {result}"
    );
    assert!(result.win_rate_a > result.win_rate_b, "heuristic should win more hands than random: {result}");
}

/// Same check via `play_match` directly, seats fixed - confirms the harness
/// agrees with a plain unmediated match, not just the paired comparison.
#[test]
fn heuristic_beats_random_in_a_single_plain_match_on_average() {
    let config = GameConfig::six_card_golf(2);
    let mut heuristic = HeuristicAgent;
    let mut random = RandomAgent;
    let mut decision_rng = StdRng::seed_from_u64(99);

    let mut total_heuristic = 0i64;
    let mut total_random = 0i64;
    for hand in 0..200u64 {
        let mut deal_rng = StdRng::seed_from_u64(hand);
        let scores = play_match(&config, &mut [&mut heuristic, &mut random], &mut deal_rng, &mut decision_rng);
        total_heuristic += scores[0] as i64;
        total_random += scores[1] as i64;
    }
    assert!(
        total_heuristic < total_random,
        "heuristic total {total_heuristic} should beat random total {total_random} over 200 hands"
    );
}

/// Regression test: four identical heuristics in 4-player self-play could
/// reach a deadlock where every draw looks worse than waiting to everyone at
/// once, so nobody swaps and the round never ends (seen live as an exact
/// period-3 cycle in stock/discard sizes). `PATIENCE_RESHUFFLES` fixes this;
/// the step cap here is 100x what a healthy game needs, so a regression fails
/// loudly instead of hanging the suite.
#[test]
fn heuristic_self_play_terminates_even_in_a_four_player_deadlock() {
    let config = GameConfig { grid_rows: 3, grid_cols: 3, reveal_count: 3, ..GameConfig::six_card_golf(4) };
    let space = ActionSpace::new(&config);
    let mut agent = HeuristicAgent;

    for seed in 0..40u64 {
        let mut deal_rng = StdRng::seed_from_u64(seed);
        let mut decision_rng = StdRng::seed_from_u64(seed ^ 0xABCD);
        let mut state = GameState::new_with_rng(config.clone(), &mut deal_rng).unwrap();

        let mut steps = 0u64;
        while let Some(obs) = Observation::for_actor(&state) {
            let index = agent.act(&obs, &space, &mut decision_rng);
            let action = space.action_at(index, &obs.turn).unwrap();
            state.apply(action).unwrap();
            steps += 1;
            assert!(steps < 20_000, "seed {seed}: heuristic self-play did not terminate");
        }
    }
}
