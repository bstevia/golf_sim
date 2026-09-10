use std::fmt;

/// Rules for a single hand of Golf; one deck, 2x3 grid, 2 revealed
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameConfig {
    pub num_players: usize,
    pub num_decks: usize,
    pub grid_rows: usize,
    pub grid_cols: usize,
    pub reveal_count: usize,
    pub final_turns_for_others: bool,
}

impl GameConfig {
    pub fn six_card_golf(num_players: usize) -> Self {
        GameConfig {
            num_players,
            num_decks: 1,
            grid_rows: 2,
            grid_cols: 3,
            reveal_count: 2,
            final_turns_for_others: true,
        }
    }

    pub fn cards_per_player(&self) -> usize {
        self.grid_rows * self.grid_cols
    }

    pub fn deck_size(&self) -> usize {
        52 * self.num_decks
    }

    /// Cards needed just to deal every grid plus one to start the discard
    /// pile - the minimum for the round to even begin.
    pub fn cards_needed(&self) -> usize {
        self.num_players * self.cards_per_player() + 1
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.num_players == 0 {
            return Err(ConfigError::NoPlayers);
        }
        if self.num_decks == 0 {
            return Err(ConfigError::NoDecks);
        }
        if self.grid_rows == 0 || self.grid_cols == 0 {
            return Err(ConfigError::EmptyGrid);
        }
        if self.reveal_count > self.cards_per_player() {
            return Err(ConfigError::RevealTooLarge {
                reveal_count: self.reveal_count,
                grid_size: self.cards_per_player(),
            });
        }
        if self.cards_needed() > self.deck_size() {
            return Err(ConfigError::NotEnoughCards {
                needed: self.cards_needed(),
                available: self.deck_size(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    NoPlayers,
    NoDecks,
    EmptyGrid,
    RevealTooLarge { reveal_count: usize, grid_size: usize },
    NotEnoughCards { needed: usize, available: usize },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ConfigError::NoPlayers => write!(f, "need at least one player"),
            ConfigError::NoDecks => write!(f, "need at least one deck"),
            ConfigError::EmptyGrid => write!(f, "grid must have at least one row and one column"),
            ConfigError::RevealTooLarge { reveal_count, grid_size } => write!(
                f,
                "reveal_count ({reveal_count}) can't exceed the grid size ({grid_size})"
            ),
            ConfigError::NotEnoughCards { needed, available } => write!(
                f,
                "need {needed} cards to deal this config but only {available} are in the shoe \
                 (raise --decks or lower --players/--grid)"
            ),
        }
    }
}

impl std::error::Error for ConfigError {}
