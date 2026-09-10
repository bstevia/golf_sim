#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawSource {
    Stock,
    Discard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Reveal(usize),
    Draw(DrawSource),
    /// Swap the just-drawn card into grid cell `index`; the card that was there goes face up onto the discard pile
    Swap(usize),
    DiscardDrawn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionError {
    WrongPhase,
    CellOutOfRange { index: usize, grid_size: usize },
    CellAlreadyFaceUp { index: usize },
    StockEmpty,
}

impl std::fmt::Display for ActionError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ActionError::WrongPhase => write!(f, "action is not legal in the current phase"),
            ActionError::CellOutOfRange { index, grid_size } => {
                write!(f, "cell {index} is out of range for a {grid_size}-cell grid")
            }
            ActionError::CellAlreadyFaceUp { index } => {
                write!(f, "cell {index} is already face up")
            }
            ActionError::StockEmpty => write!(f, "no cards left to draw from the stock"),
        }
    }
}

impl std::error::Error for ActionError {}
