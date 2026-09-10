use super::grid::{Cell, PlayerGrid};


pub fn score_grid(grid: &PlayerGrid) -> i32 {
    grid.columns().map(|column| score_column(&column)).sum()
}

fn score_column(column: &[Cell]) -> i32 {
    let ranks: Vec<_> = column.iter().map(|c| c.card().rank).collect();
    if ranks.len() > 1 && ranks.iter().all(|r| *r == ranks[0]) {
        return 0;
    }
    column.iter().map(|c| c.card().rank.golf_value()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::card::{Card, Rank, Suit};

    #[allow(clippy::needless_range_loop)]
    fn grid_of(ranks: [[Rank; 2]; 3]) -> PlayerGrid {
        // 2 rows x 3 cols, row-major. `ranks[col][row]` is clearest as
        // explicit indices rather than an iterator adapter here.
        let mut cards = Vec::with_capacity(6);
        for row in 0..2 {
            for col in 0..3 {
                cards.push(Card::new(ranks[col][row], Suit::Spades));
            }
        }
        let mut grid = PlayerGrid::new(2, 3, cards);
        for i in 0..6 {
            grid.reveal(i);
        }
        grid
    }

    #[test]
    fn matching_column_scores_zero_even_for_kings_and_twos() {
        let grid = grid_of([
            [Rank::King, Rank::King], // 0 (already 0 each, still a match)
            [Rank::Two, Rank::Two],   // would be -4 unmatched, 0 matched
            [Rank::Nine, Rank::Five], // 9 + 5 = 14
        ]);
        assert_eq!(score_grid(&grid), 14);
    }

    #[test]
    fn unmatched_column_sums_golf_values() {
        let grid = grid_of([
            [Rank::Ace, Rank::King],   // 1 + 0
            [Rank::Two, Rank::Three],  // -2 + 3
            [Rank::Jack, Rank::Queen], // 10 + 10
        ]);
        let expected: i32 = 1 - 2 + 3 + 10 + 10; // Ace + King(0) + Two + Three + Jack + Queen
        assert_eq!(score_grid(&grid), expected);
    }
}
