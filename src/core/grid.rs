use super::card::Card;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    FaceDown(Card),
    FaceUp(Card),
}

impl Cell {
    pub fn card(&self) -> Card {
        match self {
            Cell::FaceDown(c) | Cell::FaceUp(c) => *c,
        }
    }

    pub fn is_face_up(&self) -> bool {
        matches!(self, Cell::FaceUp(_))
    }
}

/// One player's layout: `rows` x `cols` cells, row-major.
#[derive(Debug, Clone)]
pub struct PlayerGrid {
    rows: usize,
    cols: usize,
    cells: Vec<Cell>,
}

impl PlayerGrid {
    pub fn new(rows: usize, cols: usize, cards: Vec<Card>) -> Self {
        assert_eq!(
            cards.len(),
            rows * cols,
            "dealt {} cards for a {rows}x{cols} grid",
            cards.len()
        );
        PlayerGrid {
            rows,
            cols,
            cells: cards.into_iter().map(Cell::FaceDown).collect(),
        }
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

    pub fn cell(&self, index: usize) -> Cell {
        self.cells[index]
    }

    pub fn face_down_indices(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.cells.len()).filter(move |&i| !self.cells[i].is_face_up())
    }

    /// Turns a still-face-down cell face up in place, without changing the
    /// card. Used for the initial reveal.
    pub fn reveal(&mut self, index: usize) {
        self.cells[index] = Cell::FaceUp(self.cells[index].card());
    }

    /// Swaps in `new_card` face up at `index`, returning the cell that was
    /// there (its card goes to the discard pile face up regardless of
    /// whether it was already revealed).
    pub fn replace(&mut self, index: usize, new_card: Card) -> Cell {
        std::mem::replace(&mut self.cells[index], Cell::FaceUp(new_card))
    }

    pub fn is_all_face_up(&self) -> bool {
        self.cells.iter().all(Cell::is_face_up)
    }

    pub fn column(&self, col: usize) -> impl Iterator<Item = Cell> + '_ {
        (0..self.rows).map(move |r| self.cells[r * self.cols + col])
    }

    pub fn columns(&self) -> impl Iterator<Item = Vec<Cell>> + '_ {
        (0..self.cols).map(move |c| self.column(c).collect())
    }
}
