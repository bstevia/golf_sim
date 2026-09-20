use crate::core::action::{Action, DrawSource};
use crate::core::config::GameConfig;
use crate::repr::observation::{Observation, TurnView};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionSpace {
    grid_size: usize,
}

impl ActionSpace {
    pub fn new(config: &GameConfig) -> Self {
        ActionSpace { grid_size: config.cards_per_player() }
    }

    pub fn len(&self) -> usize {
        self.grid_size + 3
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn grid_size(&self) -> usize {
        self.grid_size
    }

    pub fn discard_drawn(&self) -> usize {
        self.grid_size
    }

    pub fn draw_stock(&self) -> usize {
        self.grid_size + 1
    }

    pub fn draw_discard(&self) -> usize {
        self.grid_size + 2
    }

    pub fn index_of(&self, action: Action) -> usize {
        match action {
            Action::Reveal(i) | Action::Swap(i) => i,
            Action::DiscardDrawn => self.discard_drawn(),
            Action::Draw(DrawSource::Stock) => self.draw_stock(),
            Action::Draw(DrawSource::Discard) => self.draw_discard(),
        }
    }

    /// Resolves a flat index back to an action. Needs the phase, since cell
    /// slots are overloaded. Indices are in whatever frame `turn` came from -
    /// map through a canonicalization permutation first if you used one.
    pub fn action_at(&self, index: usize, turn: &TurnView) -> Option<Action> {
        if index < self.grid_size {
            return match turn {
                TurnView::Reveal { .. } => Some(Action::Reveal(index)),
                TurnView::Decide { .. } => Some(Action::Swap(index)),
                TurnView::Draw => None,
            };
        }
        match (index - self.grid_size, turn) {
            (0, TurnView::Decide { .. }) => Some(Action::DiscardDrawn),
            (1, TurnView::Draw) => Some(Action::Draw(DrawSource::Stock)),
            (2, TurnView::Draw) => Some(Action::Draw(DrawSource::Discard)),
            _ => None,
        }
    }

    /// Legality mask over the flat space; agrees with `GameState::legal_actions`.
    pub fn mask(&self, obs: &Observation) -> Vec<bool> {
        let mut mask = vec![false; self.len()];
        match obs.turn {
            TurnView::Reveal { .. } => {
                let grid = obs.own_grid();
                for (i, slot) in mask.iter_mut().enumerate().take(grid.len()) {
                    *slot = !grid.cell(i).is_known();
                }
            }
            TurnView::Draw => {
                mask[self.draw_stock()] = true;
                mask[self.draw_discard()] = obs.discard_top.is_some();
            }
            TurnView::Decide { .. } => {
                for slot in mask.iter_mut().take(self.grid_size) {
                    *slot = true;
                }
                mask[self.discard_drawn()] = true;
            }
        }
        mask
    }
}
