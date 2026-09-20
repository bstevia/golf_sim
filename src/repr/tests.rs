use super::*;
use crate::core::action::{Action, DrawSource};
use crate::core::card::Rank;
use crate::core::config::GameConfig;
use crate::core::state::GameState;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::SeedableRng;

fn grid_view(rows: usize, cols: usize, cells: &[CellView]) -> GridView {
    GridView::from_cells(rows, cols, cells)
}

#[test]
fn canonicalize_collapses_grids_that_differ_only_by_column_order() {
    let up = CellView::Up;
    let a = &mut grid_view(
        2,
        3,
        &[up(Rank::Four), up(Rank::Nine), CellView::Unknown,
          up(Rank::Four), up(Rank::Two), CellView::Unknown],
    );
    // same columns, shuffled and one flipped
    let b = &mut grid_view(
        2,
        3,
        &[CellView::Unknown, up(Rank::Two), up(Rank::Four),
          CellView::Unknown, up(Rank::Nine), up(Rank::Four)],
    );
    a.canonicalize();
    b.canonicalize();
    assert_eq!(a, b, "column and row-within-column order must not survive canonicalization");
}

#[test]
fn canonicalize_permutation_maps_cells_back_to_their_original_slots() {
    let mut rng = StdRng::seed_from_u64(11);
    let mut state = GameState::new_with_rng(GameConfig::six_card_golf(3), &mut rng).unwrap();
    for _ in 0..6 {
        let action = *state.legal_actions().choose(&mut rng).unwrap();
        state.apply(action).unwrap();
    }

    let original = Observation::for_actor(&state).unwrap();
    let mut canonical = original.clone();
    let perm = canonical.canonicalize();

    assert_eq!(perm.len(), original.own_grid().len());
    let mut seen = perm.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), perm.len(), "permutation must be a bijection");

    for (canonical_index, &original_index) in perm.iter().enumerate() {
        assert_eq!(
            canonical.own_grid().cell(canonical_index),
            original.own_grid().cell(original_index)
        );
    }
}

#[test]
fn observation_never_reveals_a_face_down_card() {
    let mut rng = StdRng::seed_from_u64(3);
    let mut state = GameState::new_with_rng(GameConfig::six_card_golf(2), &mut rng).unwrap();
    while !state.is_round_over() {
        let obs = Observation::for_actor(&state).unwrap();
        for (offset, view) in obs.grids.iter().enumerate() {
            let real = &state.grids[(obs.seat + offset) % 2];
            for i in 0..real.len() {
                match view.cell(i) {
                    CellView::Up(rank) => {
                        assert!(real.cell(i).is_face_up());
                        assert_eq!(rank, real.cell(i).card().rank);
                    }
                    CellView::Unknown => assert!(!real.cell(i).is_face_up()),
                }
            }
        }
        let action = *state.legal_actions().choose(&mut rng).unwrap();
        state.apply(action).unwrap();
    }
}

/// Every card is either known (face up, discarded, or in hand) or unseen
/// (face down or in the stock) - the two counts must add up.
#[test]
fn unseen_counts_account_for_exactly_the_hidden_cards() {
    for seed in 0..25u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut state = GameState::new_with_rng(GameConfig::six_card_golf(4), &mut rng).unwrap();
        while !state.is_round_over() {
            let obs = Observation::for_actor(&state).unwrap();
            let face_down: usize = obs.grids.iter().map(GridView::face_down_count).sum();
            assert_eq!(
                obs.unseen_total(),
                face_down + obs.stock_remaining,
                "seed {seed}: unseen pool must be the face-down cells plus the stock"
            );
            let action = *state.legal_actions().choose(&mut rng).unwrap();
            state.apply(action).unwrap();
        }
    }
}

#[test]
fn action_mask_agrees_with_the_engine_in_every_phase() {
    for seed in 0..25u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let config = GameConfig::six_card_golf(3);
        let space = ActionSpace::new(&config);
        let mut state = GameState::new_with_rng(config, &mut rng).unwrap();

        while !state.is_round_over() {
            let obs = Observation::for_actor(&state).unwrap();
            let mask = space.mask(&obs);
            let legal = state.legal_actions();

            for action in &legal {
                assert!(
                    mask[space.index_of(*action)],
                    "seed {seed}: engine allows {action:?} but the mask does not"
                );
            }
            let allowed = mask.iter().filter(|&&m| m).count();
            assert_eq!(allowed, legal.len(), "seed {seed}: mask and engine disagree on count");

            for (index, &is_legal) in mask.iter().enumerate() {
                if is_legal {
                    let action = space
                        .action_at(index, &obs.turn)
                        .expect("a masked-in index must resolve to an action");
                    assert!(legal.contains(&action), "seed {seed}: {action:?} is not actually legal");
                }
            }

            let action = *legal.choose(&mut rng).unwrap();
            state.apply(action).unwrap();
        }
    }
}

/// A cell chosen in canonical space must map back to a legal engine action.
#[test]
fn canonical_cell_choices_map_back_to_legal_engine_actions() {
    for seed in 0..15u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let config = GameConfig::six_card_golf(2);
        let space = ActionSpace::new(&config);
        let mut state = GameState::new_with_rng(config, &mut rng).unwrap();

        while !state.is_round_over() {
            let mut obs = Observation::for_actor(&state).unwrap();
            let perm = obs.canonicalize();
            let mask = space.mask(&obs);
            let legal = state.legal_actions();

            let choice = mask.iter().position(|&m| m).unwrap();
            let canonical_action = space.action_at(choice, &obs.turn).unwrap();
            let engine_action = match canonical_action {
                Action::Reveal(i) => Action::Reveal(perm[i]),
                Action::Swap(i) => Action::Swap(perm[i]),
                other => other,
            };
            assert!(legal.contains(&engine_action), "seed {seed}: {engine_action:?} is illegal");

            let action = *legal.choose(&mut rng).unwrap();
            state.apply(action).unwrap();
        }
    }
}

#[test]
fn encoder_fills_exactly_its_declared_width() {
    let config = GameConfig::six_card_golf(2);
    let encoder = Encoder::new(&config);
    assert_eq!(encoder.len(), 2 * (6 * 14 + 3) + 49 + 2);

    let mut rng = StdRng::seed_from_u64(5);
    let mut state = GameState::new_with_rng(config, &mut rng).unwrap();
    let mut buffer = vec![0.0; encoder.len()];
    while !state.is_round_over() {
        let obs = Observation::for_actor(&state).unwrap();
        encoder.encode_into(&obs, &mut buffer);
        assert_eq!(encoder.encode(&obs), buffer);
        assert!(buffer.iter().all(|f| f.is_finite()), "encoded a non-finite feature");
        let action = *state.legal_actions().choose(&mut rng).unwrap();
        state.apply(action).unwrap();
    }
}

#[test]
fn every_cell_gets_exactly_one_hot_slot() {
    let config = GameConfig::six_card_golf(2);
    let encoder = Encoder::new(&config);
    let mut rng = StdRng::seed_from_u64(8);
    let mut state = GameState::new_with_rng(config, &mut rng).unwrap();
    for _ in 0..4 {
        let action = *state.legal_actions().choose(&mut rng).unwrap();
        state.apply(action).unwrap();
    }
    let obs = Observation::for_actor(&state).unwrap();
    let features = encoder.encode(&obs);
    let stride = 6 * 14 + 3; // one grid's cell one-hots + its 3 derived features
    for grid in 0..2 {
        for cell in 0..6 {
            let at = grid * stride + cell * 14;
            let block: f32 = features[at..at + 14].iter().sum();
            assert_eq!(block, 1.0, "grid {grid} cell {cell} must have exactly one slot set");
        }
    }
}

#[test]
fn seat_rotation_puts_the_observer_first_and_places_the_knocker_relatively() {
    let mut rng = StdRng::seed_from_u64(21);
    let mut state = GameState::new_with_rng(GameConfig::six_card_golf(4), &mut rng).unwrap();
    while !state.is_round_over() {
        let obs = Observation::for_actor(&state).unwrap();
        assert_eq!(obs.grids.len(), 4);
        if let (Some(relative), Some(absolute)) = (obs.knocked_by, state.knocked_by) {
            assert_eq!((obs.seat + relative) % 4, absolute);
        }
        let action = *state.legal_actions().choose(&mut rng).unwrap();
        state.apply(action).unwrap();
    }
}

#[test]
fn expected_unseen_value_sits_between_the_best_and_worst_card() {
    let mut rng = StdRng::seed_from_u64(13);
    let state = GameState::new_with_rng(GameConfig::six_card_golf(2), &mut rng).unwrap();
    let obs = Observation::for_actor(&state).unwrap();
    let expected = obs.expected_unseen_value();
    assert!((-2.0..=10.0).contains(&expected), "expected value {expected} is out of range");
    let fresh: f32 = Rank::ALL.iter().map(|r| 4 * r.golf_value()).sum::<i32>() as f32 / 52.0;
    assert!((expected - fresh).abs() < 1.5, "{expected} should be near the deck mean {fresh}");
}

#[test]
fn drawn_card_is_excluded_from_the_unseen_pool() {
    let mut rng = StdRng::seed_from_u64(17);
    let mut state = GameState::new_with_rng(GameConfig::six_card_golf(2), &mut rng).unwrap();
    while !matches!(state.phase, crate::core::state::Phase::AwaitingDraw { .. }) {
        let action = *state.legal_actions().choose(&mut rng).unwrap();
        state.apply(action).unwrap();
    }
    state.apply(Action::Draw(DrawSource::Stock)).unwrap();

    let obs = Observation::for_actor(&state).unwrap();
    let TurnView::Decide { drawn, source } = obs.turn else {
        panic!("expected a decision phase");
    };
    assert_eq!(source, DrawSource::Stock);
    let face_down: usize = obs.grids.iter().map(GridView::face_down_count).sum();
    assert_eq!(obs.unseen_total(), face_down + obs.stock_remaining);
    let _ = drawn;
}

/// Sweeps grid shapes and player counts the simulator advertises, to catch
/// index-arithmetic bugs that only show up off the default 2x3 shape.
#[test]
fn representation_holds_up_across_every_advertised_config() {
    let configs = [
        GameConfig::six_card_golf(2),
        GameConfig { grid_rows: 2, grid_cols: 2, reveal_count: 1, ..GameConfig::six_card_golf(3) },
        GameConfig { grid_rows: 3, grid_cols: 3, reveal_count: 3, ..GameConfig::six_card_golf(4) },
        GameConfig { grid_rows: 1, grid_cols: 4, reveal_count: 1, ..GameConfig::six_card_golf(2) },
        GameConfig { num_decks: 2, ..GameConfig::six_card_golf(6) },
        GameConfig { reveal_count: 0, ..GameConfig::six_card_golf(2) },
        GameConfig { final_turns_for_others: false, ..GameConfig::six_card_golf(3) },
    ];

    for config in configs {
        let space = ActionSpace::new(&config);
        let encoder = Encoder::new(&config);
        let mut buffer = vec![0.0; encoder.len()];

        for seed in 0..5u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut state = GameState::new_with_rng(config.clone(), &mut rng).unwrap();
            while !state.is_round_over() {
                let mut obs = Observation::for_actor(&state).unwrap();
                let legal = state.legal_actions();

                assert_eq!(space.mask(&obs).iter().filter(|&&m| m).count(), legal.len());
                let face_down: usize = obs.grids.iter().map(GridView::face_down_count).sum();
                assert_eq!(obs.unseen_total(), face_down + obs.stock_remaining);

                encoder.encode_into(&obs, &mut buffer);
                assert!(buffer.iter().all(|f| f.is_finite()));

                if config.grid_rows == 1 {
                    assert_eq!(obs.own_grid().matched_columns(), 0);
                }

                let perm = obs.canonicalize();
                encoder.encode_into(&obs, &mut buffer);
                assert_eq!(perm.len(), config.cards_per_player());

                let action = *legal.choose(&mut rng).unwrap();
                state.apply(action).unwrap();
            }
        }
    }
}
