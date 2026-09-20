use super::Agent;
use crate::core::card::Rank;
use crate::repr::{rank_index, ActionSpace, CellView, GridView, Observation, TurnView, NUM_RANKS};
use rand::RngCore;

#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicAgent;

impl Agent for HeuristicAgent {
    fn act(&mut self, obs: &Observation, space: &ActionSpace, _rng: &mut dyn RngCore) -> usize {
        match obs.turn {
            TurnView::Reveal { .. } => choose_reveal(obs.own_grid()),

            // The discard is already known, so it's only worth drawing if it
            // improves some cell right now.
            TurnView::Draw => match obs.discard_top {
                Some(top) => {
                    let unseen_ev = obs.expected_unseen_value();
                    let (_, delta) = best_swap_delta(obs.own_grid(), &obs.unseen, unseen_ev, top);
                    if delta < 0.0 { space.draw_discard() } else { space.draw_stock() }
                }
                None => space.draw_stock(),
            },

            TurnView::Decide { drawn, .. } => {
                let grid = obs.own_grid();
                let unseen_ev = obs.expected_unseen_value();
                let (best_index, delta) = best_swap_delta(grid, &obs.unseen, unseen_ev, drawn);
                if delta < 0.0 {
                    best_index
                } else if obs.turns >= PATIENCE_TURNS {
                    force_progress(grid, &obs.unseen, unseen_ev, drawn).unwrap_or_else(|| space.discard_drawn())
                } else {
                    space.discard_drawn()
                }
            }
        }
    }

    fn name(&self) -> &str {
        "heuristic"
    }
}

/// Reveals the face-down cell in the column with fewest face-up cells so far
/// (ties: lowest column, then row). Spreads opening reveals across columns.
pub(super) fn choose_reveal(grid: &GridView) -> usize {
    let cols = grid.cols();
    (0..grid.len())
        .filter(|&i| !grid.cell(i).is_known())
        .min_by_key(|&i| {
            let col = i % cols;
            let revealed_in_col = grid.column(col).filter(|c| c.is_known()).count();
            (revealed_in_col, col, i)
        })
        .expect("reveal phase always offers at least one face-down cell")
}

/// Guards against a real deadlock: four identical greedy agents can all find
/// every draw worse than waiting at once, so nobody swaps and the round never
/// ends. `turns` is public and strictly increasing every turn (unlike
/// `reshuffles`, which can stay at zero forever if nobody happens to draw
/// from the stock), so once it passes this, everyone forces progress on
/// their own next turn instead of discarding forever.
const PATIENCE_TURNS: u64 = 300;

/// Best cell to place `drawn` into, and the resulting change in that column's
/// expected value (negative = improvement). Ties favor the lowest index.
fn best_swap_delta(grid: &GridView, unseen: &[u16; NUM_RANKS], unseen_ev: f32, drawn: Rank) -> (usize, f32) {
    best_swap_delta_over(grid, unseen, unseen_ev, drawn, 0..grid.len())
        .expect("a grid always has at least one cell")
}

/// Give up looking for a good trade and resolve one of the grid's own
/// face-down cells instead. `None` means the grid is already fully revealed.
fn force_progress(grid: &GridView, unseen: &[u16; NUM_RANKS], unseen_ev: f32, drawn: Rank) -> Option<usize> {
    let unknown_cells = (0..grid.len()).filter(|&i| !grid.cell(i).is_known());
    best_swap_delta_over(grid, unseen, unseen_ev, drawn, unknown_cells).map(|(index, _)| index)
}

fn best_swap_delta_over(
    grid: &GridView,
    unseen: &[u16; NUM_RANKS],
    unseen_ev: f32,
    drawn: Rank,
    candidates: impl Iterator<Item = usize>,
) -> Option<(usize, f32)> {
    let cols = grid.cols();
    let mut best: Option<(usize, f32)> = None;
    for i in candidates {
        let col = i % cols;
        let row = i / cols;
        let mut column: Vec<CellView> = grid.column(col).collect();
        let before = expected_column_score(&column, unseen, unseen_ev);
        column[row] = CellView::Up(drawn);
        let after = expected_column_score(&column, unseen, unseen_ev);
        let delta = after - before;
        if best.is_none_or(|(_, best_delta)| delta < best_delta) {
            best = Some((i, delta));
        }
    }
    best
}

/// Expected golf value of a column, under the pooled belief that every
/// hidden cell and stock card is equally likely to be any unseen rank.
/// A still-possible match (every known cell so far agrees) is priced exactly
/// for one remaining unknown, and approximately (independent draws) for more
/// than one. Treats a column with zero known cells as unable to match at
/// all, which slightly underrates it - a safe, conservative bias.
fn expected_column_score(cells: &[CellView], unseen: &[u16; NUM_RANKS], unseen_ev: f32) -> f32 {
    let knowns: Vec<Rank> = cells.iter().filter_map(|c| c.rank()).collect();
    let unknown_count = cells.len() - knowns.len();

    if unknown_count == 0 {
        return if cells.len() > 1 && knowns.iter().all(|&r| r == knowns[0]) {
            0.0
        } else {
            knowns.iter().map(|r| r.golf_value() as f32).sum()
        };
    }

    let known_value: f32 = knowns.iter().map(|r| r.golf_value() as f32).sum();
    let all_knowns_agree = !knowns.is_empty() && knowns.windows(2).all(|w| w[0] == w[1]);

    if all_knowns_agree {
        let total: u32 = unseen.iter().map(|&n| n as u32).sum();
        let p_match_each = if total == 0 {
            0.0
        } else {
            unseen[rank_index(knowns[0])] as f32 / total as f32
        };
        let p_all_match = p_match_each.powi(unknown_count as i32);
        let non_match_value = known_value + unknown_count as f32 * unseen_ev;
        (1.0 - p_all_match) * non_match_value
    } else {
        known_value + unknown_count as f32 * unseen_ev
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::card::Rank;

    fn unseen_uniform(copies: u16) -> [u16; NUM_RANKS] {
        [copies; NUM_RANKS]
    }

    #[test]
    fn fully_known_matched_column_scores_zero() {
        let cells = [CellView::Up(Rank::King), CellView::Up(Rank::King)];
        assert_eq!(expected_column_score(&cells, &unseen_uniform(4), 5.0), 0.0);
    }

    #[test]
    fn fully_known_unmatched_column_sums_values() {
        let cells = [CellView::Up(Rank::Ace), CellView::Up(Rank::Three)];
        let expected = Rank::Ace.golf_value() as f32 + Rank::Three.golf_value() as f32;
        assert_eq!(expected_column_score(&cells, &unseen_uniform(4), 5.0), expected);
    }

    #[test]
    fn column_with_no_information_uses_the_unseen_average() {
        let cells = [CellView::Unknown, CellView::Unknown];
        let unseen_ev = 3.5;
        assert_eq!(expected_column_score(&cells, &unseen_uniform(4), unseen_ev), 2.0 * unseen_ev);
    }

    #[test]
    fn one_known_card_pulls_score_toward_a_possible_match() {
        let cells = [CellView::Up(Rank::King), CellView::Unknown];
        let unseen_ev = 5.0;
        let naive = Rank::King.golf_value() as f32 + unseen_ev;
        let score = expected_column_score(&cells, &unseen_uniform(4), unseen_ev);
        assert!(score < naive, "possible match should pull the expectation down below {naive}, got {score}");
        assert!(score >= 0.0, "King can only match King (0) or beat it, never go negative here");
    }

    #[test]
    fn heuristic_never_breaks_a_zero_value_match_for_no_reason() {
        // column 0: Nine/Nine match; column 1: Ten + face-down
        let grid = GridView::from_cells(
            2,
            2,
            &[
                CellView::Up(Rank::Nine),
                CellView::Up(Rank::Ten),
                CellView::Up(Rank::Nine),
                CellView::Unknown,
            ],
        );
        let (index, delta) = best_swap_delta(&grid, &unseen_uniform(4), 5.5, Rank::Five);
        assert!(delta < 0.0, "the Ten/unknown column should still improve");
        assert_ne!(index, 0, "must not swap into the matched Nine column");
        assert_ne!(index, 2, "must not swap into the matched Nine column");
    }

    #[test]
    fn heuristic_will_break_a_matched_pair_of_twos_when_it_helps() {
        // column 0: Two/Two match (worth breaking); column 1: King/King match (must not touch)
        let grid = GridView::from_cells(
            2,
            2,
            &[
                CellView::Up(Rank::Two),
                CellView::Up(Rank::King),
                CellView::Up(Rank::Two),
                CellView::Up(Rank::King),
            ],
        );
        let (index, delta) = best_swap_delta(&grid, &unseen_uniform(4), 5.5, Rank::Ace);
        assert!(delta < 0.0, "breaking the Two/Two match with an Ace should score better than 0");
        assert!(index == 0 || index == 2, "must swap into the Two/Two column, got {index}");
    }

    #[test]
    fn choose_reveal_spreads_across_columns_before_doubling_up() {
        let grid = GridView::from_cells(
            2,
            3,
            &[
                CellView::Up(Rank::Four),
                CellView::Unknown,
                CellView::Unknown,
                CellView::Unknown,
                CellView::Unknown,
                CellView::Unknown,
            ],
        );
        let choice = choose_reveal(&grid);
        let col = choice % grid.cols();
        assert_ne!(col, 0, "should prefer an untouched column over doubling up on column 0");
    }
}
