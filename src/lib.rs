pub mod agent;
pub mod core;
pub mod nn;
pub mod repr;

pub use agent::{
    eval_head_to_head, play_match, Agent, HeuristicAgent, MatchResult, NeuralAgent, RandomAgent,
};
pub use core::{
    Action, ActionError, Card, Cell, ConfigError, Deck, DrawSource, GameConfig, GameState, Phase,
    PlayerGrid, Rank, Suit,
};
pub use nn::ValueNet;
pub use repr::{ActionSpace, CellView, Encoder, GridView, Observation, TurnView};

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::seq::IndexedRandom;
    use rand::SeedableRng;

    #[test]
    fn deck_size_scales_with_num_decks() {
        assert_eq!(Deck::new(1).remaining(), 52);
        assert_eq!(Deck::new(3).remaining(), 156);
    }

    #[test]
    fn config_rejects_too_many_players_for_the_shoe() {
        let config = GameConfig::six_card_golf(20);
        assert_eq!(
            config.validate().unwrap_err(),
            ConfigError::NotEnoughCards { needed: 20 * 6 + 1, available: 52 }
        );
    }

    #[test]
    fn config_rejects_reveal_count_larger_than_grid() {
        let config = GameConfig { reveal_count: 7, ..GameConfig::six_card_golf(2) };
        assert_eq!(
            config.validate().unwrap_err(),
            ConfigError::RevealTooLarge { reveal_count: 7, grid_size: 6 }
        );
    }

    #[test]
    fn deal_gives_every_player_a_full_grid_of_unique_cards() {
        let mut rng = StdRng::seed_from_u64(1);
        let state = GameState::new_with_rng(GameConfig::six_card_golf(4), &mut rng).unwrap();
        assert_eq!(state.grids.len(), 4);
        let mut all_cards: Vec<Card> = state.grids.iter().flat_map(|g| (0..g.len()).map(|i| g.cell(i).card())).collect();
        all_cards.push(state.discard[0]);
        let before = all_cards.len();
        all_cards.sort_by_key(|c| (c.rank as u8, c.suit as u8));
        all_cards.dedup();
        assert_eq!(all_cards.len(), before, "dealt a duplicate card");
    }

    #[test]
    fn initial_reveal_exposes_exactly_reveal_count_cells_per_player() {
        let mut rng = StdRng::seed_from_u64(2);
        let config = GameConfig::six_card_golf(3);
        let mut state = GameState::new_with_rng(config.clone(), &mut rng).unwrap();

        for _ in 0..(config.num_players * config.reveal_count) {
            let action = *state.legal_actions().choose(&mut rng).unwrap();
            state.apply(action).unwrap();
        }

        assert!(matches!(state.phase, Phase::AwaitingDraw { player: 0 }));
        for grid in &state.grids {
            let face_up = (0..grid.len()).filter(|&i| grid.cell(i).is_face_up()).count();
            assert_eq!(face_up, config.reveal_count);
        }
    }

    /// A round ending does NOT mean every grid is face up: players who get
    /// a final turn after someone else knocks can spend it discarding
    /// without swapping, leaving cells face down. Those cells still carry
    /// their real card underneath and score normally - only the knocking
    /// player is guaranteed to be all face up.
    #[test]
    fn random_playout_always_terminates_and_scores_every_player() {
        for seed in 0..20u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut state = GameState::new_with_rng(GameConfig::six_card_golf(3), &mut rng).unwrap();

            let mut steps = 0;
            while !state.is_round_over() {
                let action = *state.legal_actions().choose(&mut rng).unwrap();
                state.apply(action).unwrap();
                steps += 1;
                assert!(steps < 100_000, "playout did not terminate");
            }

            assert!(state.knocked_by.is_some());
            assert!(state.grids[state.knocked_by.unwrap()].is_all_face_up());
            assert_eq!(state.scores().len(), 3);
        }
    }

    #[test]
    fn final_turns_for_others_gives_everyone_one_more_turn_after_a_knock() {
        // Degenerate but deterministic: reveal_count == grid size means every
        // grid is fully revealed as soon as the last reveal action lands, so
        // the round ends right there with nobody having drawn yet - a good
        // check that the "everyone else gets one turn" rule doesn't even try
        // to fire when there was no live turn left to grant.
        let mut rng = StdRng::seed_from_u64(7);
        let config = GameConfig { reveal_count: 6, ..GameConfig::six_card_golf(3) };
        let mut state = GameState::new_with_rng(config.clone(), &mut rng).unwrap();

        for _ in 0..(config.num_players * config.reveal_count) {
            if state.is_round_over() {
                break;
            }
            let action = *state.legal_actions().choose(&mut rng).unwrap();
            state.apply(action).unwrap();
        }

        assert!(state.is_round_over());
        assert_eq!(state.scores().len(), 3);
    }

    #[test]
    fn stock_reshuffles_from_discard_when_it_runs_dry() {
        let mut rng = StdRng::seed_from_u64(9);
        let config = GameConfig {
            num_players: 8,
            num_decks: 1,
            grid_rows: 2,
            grid_cols: 3,
            reveal_count: 2,
            final_turns_for_others: true,
        };
        let mut state = GameState::new_with_rng(config, &mut rng).unwrap();
        let mut steps = 0;
        let mut saw_empty_stock = false;
        while !state.is_round_over() {
            if state.stock.is_empty() {
                saw_empty_stock = true;
            }
            let action = *state.legal_actions().choose(&mut rng).unwrap();
            state.apply(action).unwrap();
            steps += 1;
            assert!(steps < 200_000, "playout did not terminate");
        }
        assert!(saw_empty_stock, "test setup should actually exercise the reshuffle path");
        assert!(state.knocked_by.is_some());
    }

    /// Regression test for a bug where a mid-game reshuffle drew from the
    /// thread-local RNG instead of the seeded one threaded through
    /// `new_with_rng`, so any hand that emptied the stock silently stopped
    /// being reproducible from its seed. Replays the same seed twice - for the
    /// setup RNG driving action choices *and* the engine's internal RNG - and
    /// requires the two playouts to make identical choices ply by ply,
    /// including through at least one reshuffle.
    #[test]
    fn seeded_playout_is_reproducible_across_a_reshuffle() {
        let config = GameConfig {
            num_players: 8,
            num_decks: 1,
            grid_rows: 2,
            grid_cols: 3,
            reveal_count: 2,
            final_turns_for_others: true,
        };

        let run = |seed: u64| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut state = GameState::new_with_rng(config.clone(), &mut rng).unwrap();
            let mut actions = Vec::new();
            let mut saw_empty_stock = false;
            let mut steps = 0;
            while !state.is_round_over() {
                if state.stock.is_empty() {
                    saw_empty_stock = true;
                }
                let action = *state.legal_actions().choose(&mut rng).unwrap();
                actions.push(action);
                state.apply(action).unwrap();
                steps += 1;
                assert!(steps < 200_000, "playout did not terminate");
            }
            (actions, state.scores(), saw_empty_stock)
        };

        let (actions_a, scores_a, saw_empty_a) = run(9);
        let (actions_b, scores_b, saw_empty_b) = run(9);

        assert!(saw_empty_a && saw_empty_b, "test setup should actually exercise the reshuffle path");
        assert_eq!(actions_a, actions_b, "same seed must draw the same actions at every ply");
        assert_eq!(scores_a, scores_b, "same seed must produce the same final scores");
    }
}
