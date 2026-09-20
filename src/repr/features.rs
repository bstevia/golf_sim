//! Flat `f32` encoding of an [`Observation`], for a linear model or small MLP.
//!
//! Layout, in order (`p` = players, `n` = grid cells):
//!
//! | block                                      | width        |
//! |---------------------------------------------|--------------|
//! | per grid, observer first: cell one-hots (14 wide: 13 ranks + unknown) | `p*n*14` |
//! | per grid: face-up value, face-down share, matched columns | `p*3` |
//! | phase one-hot                                | 3            |
//! | drawn rank one-hot                           | 13           |
//! | drawn source one-hot                         | 2            |
//! | discard top one-hot                          | 13           |
//! | unseen rank counts                           | 13           |
//! | expected unseen value                        | 1            |
//! | stock remaining, discard length, reshuffles  | 3            |
//! | knocked flag                                 | 1            |
//! | knocking seat one-hot (relative)             | `p`          |
//!
//! The per-grid derived block is technically redundant with the one-hots, but
//! handing it over directly saves training time.

use crate::core::action::DrawSource;
use crate::core::card::Rank;
use crate::core::config::GameConfig;
use crate::repr::observation::{rank_index, CellView, Observation, TurnView, NUM_RANKS};

const CELL_SLOTS: usize = NUM_RANKS + 1;
const DERIVED_PER_GRID: usize = 3;
/// phase 3 + drawn rank 13 + source 2 + discard top 13 + unseen 13
/// + expected unseen 1 + scalars 3 + knocked flag 1.
const GLOBAL_BLOCK: usize = 49;

#[derive(Debug, Clone)]
pub struct Encoder {
    num_players: usize,
    grid_size: usize,
    grid_cols: usize,
    copies_per_rank: f32,
    deck_size: f32,
}

impl Encoder {
    pub fn new(config: &GameConfig) -> Self {
        Encoder {
            num_players: config.num_players,
            grid_size: config.cards_per_player(),
            grid_cols: config.grid_cols,
            copies_per_rank: (4 * config.num_decks) as f32,
            deck_size: config.deck_size() as f32,
        }
    }

    pub fn len(&self) -> usize {
        self.num_players * (self.grid_size * CELL_SLOTS + DERIVED_PER_GRID)
            + GLOBAL_BLOCK
            + self.num_players
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn encode(&self, obs: &Observation) -> Vec<f32> {
        let mut out = vec![0.0; self.len()];
        self.encode_into(obs, &mut out);
        out
    }

    /// Fills `out` in place - use in a training loop to reuse one buffer.
    pub fn encode_into(&self, obs: &Observation, out: &mut [f32]) {
        assert_eq!(
            out.len(),
            self.len(),
            "encode_into needs a buffer of exactly {} floats",
            self.len()
        );
        out.fill(0.0);
        let mut at = 0;

        for grid in &obs.grids {
            for i in 0..grid.len() {
                let slot = match grid.cell(i) {
                    CellView::Up(rank) => rank_index(rank),
                    CellView::Unknown => NUM_RANKS,
                };
                out[at + i * CELL_SLOTS + slot] = 1.0;
            }
            at += self.grid_size * CELL_SLOTS;

            // 10 is the worst card value (Jack/Queen), used as the scale.
            out[at] = grid.face_up_value_sum() as f32 / (10.0 * self.grid_size as f32);
            out[at + 1] = grid.face_down_count() as f32 / self.grid_size as f32;
            out[at + 2] = grid.matched_columns() as f32 / self.grid_cols as f32;
            at += DERIVED_PER_GRID;
        }

        let phase_slot = match obs.turn {
            TurnView::Reveal { .. } => 0,
            TurnView::Draw => 1,
            TurnView::Decide { .. } => 2,
        };
        out[at + phase_slot] = 1.0;
        at += 3;

        if let TurnView::Decide { drawn, source } = obs.turn {
            out[at + rank_index(drawn)] = 1.0;
            out[at + NUM_RANKS + if source == DrawSource::Stock { 0 } else { 1 }] = 1.0;
        }
        at += NUM_RANKS + 2;

        if let Some(top) = obs.discard_top {
            out[at + rank_index(top)] = 1.0;
        }
        at += NUM_RANKS;

        for &rank in Rank::ALL.iter() {
            out[at + rank_index(rank)] =
                obs.unseen[rank_index(rank)] as f32 / self.copies_per_rank;
        }
        at += NUM_RANKS;

        out[at] = obs.expected_unseen_value() / 10.0;
        out[at + 1] = obs.stock_remaining as f32 / self.deck_size;
        out[at + 2] = obs.discard_len as f32 / self.deck_size;
        out[at + 3] = (obs.reshuffles as f32).min(4.0) / 4.0; // saturating: reshuffles are rare
        at += 4;

        if let Some(relative_seat) = obs.knocked_by {
            out[at] = 1.0;
            out[at + 1 + relative_seat] = 1.0;
        }
    }
}
