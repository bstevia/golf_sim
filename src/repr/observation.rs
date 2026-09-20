use crate::core::action::DrawSource;
use crate::core::card::Rank;
use crate::core::config::GameConfig;
use crate::core::state::{GameState, Phase};

pub const NUM_RANKS: usize = 13;

/// `Rank` runs 2..=14, so this maps it to a dense 0..13 index.
pub fn rank_index(rank: Rank) -> usize {
    rank as usize - 2
}

/// A cell as seen from outside: a revealed rank, or nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CellView {
    Up(Rank),
    Unknown,
}

impl CellView {
    pub fn rank(self) -> Option<Rank> {
        match self {
            CellView::Up(rank) => Some(rank),
            CellView::Unknown => None,
        }
    }

    pub fn is_known(self) -> bool {
        matches!(self, CellView::Up(_))
    }

    pub fn value(self) -> Option<i32> {
        self.rank().map(|r| r.golf_value())
    }
}

/// One player's grid with face-down cards masked out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridView {
    rows: usize,
    cols: usize,
    cells: Vec<CellView>,
}

impl GridView {
    #[cfg(test)]
    pub(crate) fn from_cells(rows: usize, cols: usize, cells: &[CellView]) -> GridView {
        assert_eq!(cells.len(), rows * cols);
        GridView { rows, cols, cells: cells.to_vec() }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn cell(&self, index: usize) -> CellView {
        self.cells[index]
    }

    pub fn cells(&self) -> &[CellView] {
        &self.cells
    }

    pub fn column(&self, col: usize) -> impl Iterator<Item = CellView> + '_ {
        (0..self.rows).map(move |r| self.cells[r * self.cols + col])
    }

    pub fn face_down_count(&self) -> usize {
        self.cells.iter().filter(|c| !c.is_known()).count()
    }

    /// Sum of revealed cell values, ignoring the column-match rule.
    pub fn face_up_value_sum(&self) -> i32 {
        self.cells.iter().filter_map(|c| c.value()).sum()
    }

    /// Columns already fully revealed and all one rank - locked at zero.
    pub fn matched_columns(&self) -> usize {
        if self.rows < 2 {
            return 0;
        }
        (0..self.cols)
            .filter(|&c| {
                let mut column = self.column(c);
                match column.next() {
                    Some(CellView::Up(first)) => column.all(|cell| cell == CellView::Up(first)),
                    _ => false,
                }
            })
            .count()
    }

    /// Sorts cells within each column, then sorts columns against each other.
    /// Columns and rows-within-a-column are interchangeable, so this merges
    /// layouts that are really the same position (~12x fewer for a 2x3 grid).
    ///
    /// Returns `perm[canonical_index] == original_index` - map a chosen cell
    /// back through it before sending an action to the engine.
    pub fn canonicalize(&mut self) -> Vec<usize> {
        let mut columns: Vec<(Vec<CellView>, Vec<usize>)> = (0..self.cols)
            .map(|c| {
                let mut entries: Vec<(CellView, usize)> = (0..self.rows)
                    .map(|r| {
                        let i = r * self.cols + c;
                        (self.cells[i], i)
                    })
                    .collect();
                entries.sort();
                entries.into_iter().unzip()
            })
            .collect();
        columns.sort_by(|a, b| a.0.cmp(&b.0)); // stable: keeps equal columns in order

        let mut cells = vec![CellView::Unknown; self.cells.len()];
        let mut perm = vec![0usize; self.cells.len()];
        for (c, (column_cells, column_origins)) in columns.iter().enumerate() {
            for r in 0..self.rows {
                let dest = r * self.cols + c;
                cells[dest] = column_cells[r];
                perm[dest] = column_origins[r];
            }
        }
        self.cells = cells;
        perm
    }
}

/// The decision facing the observer right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnView {
    Reveal { remaining: usize },
    Draw,
    Decide { drawn: Rank, source: DrawSource },
}

/// Everything the acting player knows, and nothing else.
///
/// Seats are rotated so `grids[0]` is the observer and `grids[i]` is the
/// player `i` seats to their left - keeps a trained policy seat-invariant.
#[derive(Debug, Clone)]
pub struct Observation {
    pub config: GameConfig,
    /// The observer's true seat, for mapping actions back to the engine.
    pub seat: usize,
    pub grids: Vec<GridView>,
    pub discard_top: Option<Rank>,
    pub discard_len: usize,
    pub stock_remaining: usize,
    /// Count of each rank the observer hasn't seen: face-down cells (incl.
    /// their own) plus the stock. See `reshuffles`.
    pub unseen: [u16; NUM_RANKS],
    /// Times the discard has been reshuffled into the stock. Once this is
    /// nonzero, `unseen` is no longer exact - the stock and face-down cells
    /// stop being truly interchangeable.
    pub reshuffles: u32,
    /// Seat of the player who went out, relative to the observer.
    pub knocked_by: Option<usize>,
    pub turn: TurnView,
}

impl Observation {
    /// Builds the view for whoever is on turn. `None` once the round is over.
    ///
    /// The only constructor: building one for a non-acting seat during
    /// `AwaitingDecision` would leak the drawn card, the one private thing in
    /// this game.
    pub fn for_actor(state: &GameState) -> Option<Observation> {
        let (seat, turn) = match &state.phase {
            Phase::AwaitingReveal { player, remaining } => {
                (*player, TurnView::Reveal { remaining: *remaining })
            }
            Phase::AwaitingDraw { player } => (*player, TurnView::Draw),
            Phase::AwaitingDecision { player, drawn, source } => {
                (*player, TurnView::Decide { drawn: drawn.rank, source: *source })
            }
            Phase::RoundOver => return None,
        };

        let num_players = state.config.num_players;
        let mut unseen = [(4 * state.config.num_decks) as u16; NUM_RANKS];

        let grids = (0..num_players)
            .map(|offset| {
                let grid = &state.grids[(seat + offset) % num_players];
                let cells = (0..grid.len())
                    .map(|i| {
                        let cell = grid.cell(i);
                        if cell.is_face_up() {
                            unseen[rank_index(cell.card().rank)] -= 1;
                            CellView::Up(cell.card().rank)
                        } else {
                            CellView::Unknown
                        }
                    })
                    .collect();
                GridView { rows: grid.rows(), cols: grid.cols(), cells }
            })
            .collect();

        for card in &state.discard {
            unseen[rank_index(card.rank)] -= 1;
        }
        // The observer is holding the drawn card, so it's neither in the stock nor on the discard.
        if let TurnView::Decide { drawn, .. } = turn {
            unseen[rank_index(drawn)] -= 1;
        }

        Some(Observation {
            config: state.config.clone(),
            seat,
            grids,
            discard_top: state.discard.last().map(|c| c.rank),
            discard_len: state.discard.len(),
            stock_remaining: state.stock.remaining(),
            unseen,
            reshuffles: state.reshuffles,
            knocked_by: state
                .knocked_by
                .map(|knocker| (knocker + num_players - seat) % num_players),
            turn,
        })
    }

    pub fn own_grid(&self) -> &GridView {
        &self.grids[0]
    }

    pub fn unseen_total(&self) -> usize {
        self.unseen.iter().map(|&n| n as usize).sum()
    }

    /// Expected golf value of a cell nobody has seen: the bar a drawn card
    /// has to beat.
    pub fn expected_unseen_value(&self) -> f32 {
        let total = self.unseen_total();
        if total == 0 {
            return 0.0;
        }
        let sum: i32 = Rank::ALL
            .iter()
            .map(|&r| self.unseen[rank_index(r)] as i32 * r.golf_value())
            .sum();
        sum as f32 / total as f32
    }

    /// Canonicalizes every grid. Returns the observer's own permutation for
    /// mapping a chosen cell back to a real engine action.
    pub fn canonicalize(&mut self) -> Vec<usize> {
        let perm = self.grids[0].canonicalize();
        for grid in &mut self.grids[1..] {
            grid.canonicalize();
        }
        perm
    }
}
