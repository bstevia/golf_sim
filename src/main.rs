use clap::Parser;
use golf_sim::{Action, DrawSource, GameConfig, GameState, Phase};
use rand::rng;
use rand::seq::IndexedRandom;
use rayon::prelude::*;

/// Simulate hands of Golf with configurable players, decks, and grid size.
/// Play is uniformly random for now - a solver comes later.
#[derive(Parser, Debug)]
#[command(version)]
struct Cli {
    #[arg(short, long, default_value_t = 2)]
    players: usize,

    #[arg(short, long, default_value_t = 1)]
    decks: usize,

    #[arg(long, default_value_t = 2)]
    rows: usize,

    #[arg(long, default_value_t = 3)]
    cols: usize,

    #[arg(short, long, default_value_t = 2)]
    reveal: usize,

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    final_turns: bool,

    #[arg(short = 'n', long, default_value_t = 100_000)]
    simulations: u64,

    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let cli = Cli::parse();

    let config = GameConfig {
        num_players: cli.players,
        num_decks: cli.decks,
        grid_rows: cli.rows,
        grid_cols: cli.cols,
        reveal_count: cli.reveal,
        final_turns_for_others: cli.final_turns,
    };

    if let Err(err) = config.validate() {
        eprintln!("invalid config: {err}");
        std::process::exit(1);
    }

    if cli.verbose {
        print_sample_hand(&config);
    }

    let totals: Vec<i64> = (0..cli.simulations)
        .into_par_iter()
        .map(|_| play_random_hand(&config))
        .fold(
            || vec![0i64; config.num_players],
            |mut acc, scores| {
                for (a, s) in acc.iter_mut().zip(scores) {
                    *a += s as i64;
                }
                acc
            },
        )
        .reduce(
            || vec![0i64; config.num_players],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        );

    println!("{} hands, {} players, {} deck(s), {}x{} grid, reveal {}", cli.simulations, config.num_players, config.num_decks, config.grid_rows, config.grid_cols, config.reveal_count);
    for (seat, total) in totals.iter().enumerate() {
        let avg = *total as f64 / cli.simulations as f64;
        println!("  seat {seat}: avg score {avg:.3}");
    }
}

fn play_random_hand(config: &GameConfig) -> Vec<i32> {
    let mut rng = rng();
    let mut state = GameState::new_with_rng(config.clone(), &mut rng).expect("validated config");
    while !state.is_round_over() {
        let action = *state.legal_actions().choose(&mut rng).expect("a phase always has legal actions");
        state.apply(action).expect("action came from legal_actions");
    }
    state.scores()
}

fn print_sample_hand(config: &GameConfig) {
    let mut rng = rng();
    let mut state = GameState::new_with_rng(config.clone(), &mut rng).expect("validated config");
    let mut turn = 0;
    while !state.is_round_over() {
        let action = *state.legal_actions().choose(&mut rng).expect("a phase always has legal actions");
        describe_action(&state, action);
        state.apply(action).expect("action came from legal_actions");
        turn += 1;
        if turn > 1000 {
            break;
        }
    }
    println!("Final scores: {:?}\n", state.scores());
}

fn describe_action(state: &GameState, action: Action) {
    match (&state.phase, action) {
        (Phase::AwaitingReveal { player, .. }, Action::Reveal(i)) => {
            println!("player {player} reveals cell {i}");
        }
        (Phase::AwaitingDraw { player }, Action::Draw(DrawSource::Stock)) => {
            println!("player {player} draws from the stock");
        }
        (Phase::AwaitingDraw { player }, Action::Draw(DrawSource::Discard)) => {
            println!("player {player} draws {} from the discard", state.discard.last().unwrap());
        }
        (Phase::AwaitingDecision { player, drawn, .. }, Action::Swap(i)) => {
            println!("player {player} swaps {drawn} into cell {i}");
        }
        (Phase::AwaitingDecision { player, drawn, .. }, Action::DiscardDrawn) => {
            println!("player {player} discards {drawn}");
        }
        _ => {}
    }
}
