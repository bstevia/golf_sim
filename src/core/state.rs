use super::action::{Action, ActionError, DrawSource};
use super::card::Card;
use super::config::{ConfigError, GameConfig};
use super::deck::Deck;
use super::grid::PlayerGrid;
use super::scoring::score_grid;
use rand::rng;
use rand::Rng;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Phase {
    /// Each player turns `reveal_count` of their own cells face up before play starts, in seat order.
    AwaitingReveal { player: usize, remaining: usize },
    AwaitingDraw { player: usize },
    AwaitingDecision { player: usize, drawn: Card, source: DrawSource },
    RoundOver,
}

#[derive(Debug, Clone)]
pub struct GameState {
    pub config: GameConfig,
    pub grids: Vec<PlayerGrid>,
    pub stock: Deck,
    pub discard: Vec<Card>,
    pub phase: Phase,
    pub knocked_by: Option<usize>,
    final_turns_remaining: Option<usize>,
}

impl GameState {
    pub fn new(config: GameConfig) -> Result<Self, ConfigError> {
        Self::new_with_rng(config, &mut rng())
    }

    pub fn new_with_rng<R: Rng>(config: GameConfig, rng: &mut R) -> Result<Self, ConfigError> {
        config.validate()?;

        let mut shoe = Deck::new(config.num_decks);
        shoe.shuffle_with(rng);

        let per_player = config.cards_per_player();
        let mut grids = Vec::with_capacity(config.num_players);
        for _ in 0..config.num_players {
            let mut cards = Vec::with_capacity(per_player);
            for _ in 0..per_player {
                cards.push(shoe.deal().expect("validated enough cards to deal grids"));
            }
            grids.push(PlayerGrid::new(config.grid_rows, config.grid_cols, cards));
        }

        let discard = vec![shoe.deal().expect("validated a card left to open discard")];

        let phase = if config.reveal_count == 0 {
            Phase::AwaitingDraw { player: 0 }
        } else {
            Phase::AwaitingReveal { player: 0, remaining: config.reveal_count }
        };

        Ok(GameState {
            config,
            grids,
            stock: shoe,
            discard,
            phase,
            knocked_by: None,
            final_turns_remaining: None,
        })
    }

    /// Handles the edge case where `reveal_count == cards_per_player`: every
    /// grid is fully revealed before any turn is possible, so the initial
    /// reveal already ends the round.
    fn end_reveal_phase_if_degenerate(&mut self) {
        if self.config.reveal_count == self.config.cards_per_player()
            && self.grids.iter().any(PlayerGrid::is_all_face_up)
        {
            self.phase = Phase::RoundOver;
        }
    }

    pub fn is_round_over(&self) -> bool {
        matches!(self.phase, Phase::RoundOver)
    }

    /// Final per-player scores. Only meaningful once the round is over;
    /// scores cells by their current face (real play never calls this with
    /// face-down cells remaining, since `RoundOver` implies full reveal
    /// only for the knocking sequence - callers should check
    /// `is_round_over()` first).
    pub fn scores(&self) -> Vec<i32> {
        self.grids.iter().map(score_grid).collect()
    }

    pub fn legal_actions(&self) -> Vec<Action> {
        match &self.phase {
            Phase::AwaitingReveal { player, .. } => self.grids[*player]
                .face_down_indices()
                .map(Action::Reveal)
                .collect(),
            Phase::AwaitingDraw { .. } => {
                let mut actions = vec![Action::Draw(DrawSource::Stock)];
                if !self.discard.is_empty() {
                    actions.push(Action::Draw(DrawSource::Discard));
                }
                actions
            }
            Phase::AwaitingDecision { player, .. } => {
                let mut actions: Vec<Action> = (0..self.grids[*player].len()).map(Action::Swap).collect();
                actions.push(Action::DiscardDrawn);
                actions
            }
            Phase::RoundOver => Vec::new(),
        }
    }

    pub fn apply(&mut self, action: Action) -> Result<(), ActionError> {
        match (self.phase.clone(), action) {
            (Phase::AwaitingReveal { player, remaining }, Action::Reveal(index)) => {
                let grid = &mut self.grids[player];
                if index >= grid.len() {
                    return Err(ActionError::CellOutOfRange { index, grid_size: grid.len() });
                }
                if grid.cell(index).is_face_up() {
                    return Err(ActionError::CellAlreadyFaceUp { index });
                }
                grid.reveal(index);

                if remaining > 1 {
                    self.phase = Phase::AwaitingReveal { player, remaining: remaining - 1 };
                } else if player + 1 < self.config.num_players {
                    self.phase = Phase::AwaitingReveal {
                        player: player + 1,
                        remaining: self.config.reveal_count,
                    };
                } else {
                    self.phase = Phase::AwaitingDraw { player: 0 };
                    self.end_reveal_phase_if_degenerate();
                }
                Ok(())
            }

            (Phase::AwaitingDraw { player }, Action::Draw(source)) => {
                let drawn = match source {
                    DrawSource::Stock => self.draw_from_stock()?,
                    DrawSource::Discard => match self.discard.pop() {
                        Some(card) => card,
                        None => return Err(ActionError::WrongPhase),
                    },
                };
                self.phase = Phase::AwaitingDecision { player, drawn, source };
                Ok(())
            }

            (Phase::AwaitingDecision { player, drawn, .. }, Action::Swap(index)) => {
                let grid = &mut self.grids[player];
                if index >= grid.len() {
                    return Err(ActionError::CellOutOfRange { index, grid_size: grid.len() });
                }
                let displaced = grid.replace(index, drawn);
                self.discard.push(displaced.card());
                self.complete_turn(player);
                Ok(())
            }

            (Phase::AwaitingDecision { player, drawn, .. }, Action::DiscardDrawn) => {
                self.discard.push(drawn);
                self.complete_turn(player);
                Ok(())
            }

            _ => Err(ActionError::WrongPhase),
        }
    }

    fn draw_from_stock(&mut self) -> Result<Card, ActionError> {
        if self.stock.is_empty() {
            if self.discard.len() <= 1 {
                return Err(ActionError::StockEmpty);
            }
            let top = self.discard.pop().expect("checked non-empty above");
            let rest = std::mem::take(&mut self.discard);
            self.stock.refill_and_shuffle(rest);
            self.discard.push(top);
        }
        Ok(self.stock.deal().expect("refilled or already had cards"))
    }

    fn complete_turn(&mut self, player_who_acted: usize) {
        let already_in_final_turns = self.knocked_by.is_some();

        if !already_in_final_turns && self.grids[player_who_acted].is_all_face_up() {
            self.knocked_by = Some(player_who_acted);
            if !self.config.final_turns_for_others {
                self.phase = Phase::RoundOver;
                return;
            }
            let remaining = self.config.num_players - 1;
            if remaining == 0 {
                self.phase = Phase::RoundOver;
                return;
            }
            self.final_turns_remaining = Some(remaining);
        } else if already_in_final_turns {
            let remaining = self.final_turns_remaining.expect("set when knocked_by was set") - 1;
            if remaining == 0 {
                self.phase = Phase::RoundOver;
                return;
            }
            self.final_turns_remaining = Some(remaining);
        }

        let next = (player_who_acted + 1) % self.config.num_players;
        self.phase = Phase::AwaitingDraw { player: next };
    }
}
