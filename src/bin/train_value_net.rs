//! Trains `ValueNet` by self-play Monte Carlo regression, then saves it.
//!
//! Iteration 0 bootstraps from `HeuristicAgent` self-play, since an
//! untrained network's own choices are close to noise and would otherwise
//! generate a training distribution close to random play. Every later
//! iteration self-plays with the network trained so far (`NeuralAgent`, with
//! some exploration so it doesn't only ever see its own current opinion).
//!
//! Every visited position becomes one training example: its encoded
//! features (in the value framing - see `nn::observation_for_value`) paired
//! with the final score the acting seat actually got that hand.
//!
//! Run with `cargo run --release --bin train_value_net` - `--release`
//! matters a lot here, candle's CPU backend is much faster optimized.

use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use clap::Parser;
use golf_sim::{
    eval_head_to_head, Agent, Encoder, GameConfig, GameState, HeuristicAgent, NeuralAgent,
    Observation, RandomAgent, ValueNet,
};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

#[derive(Parser, Debug)]
struct Cli {
    /// Self-play hands generated per iteration.
    #[arg(long, default_value_t = 300)]
    games_per_iteration: u64,

    /// Training iterations (each: generate data, then fit).
    #[arg(long, default_value_t = 10)]
    iterations: u32,

    /// Passes over the dataset per iteration.
    #[arg(long, default_value_t = 5)]
    epochs_per_iteration: u32,

    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f64,

    /// Chance of a random action instead of the network's choice, during
    /// self-play data generation only (never during evaluation).
    #[arg(long, default_value_t = 0.15)]
    exploration: f32,

    /// Hands per side in each `eval_head_to_head` check.
    #[arg(long, default_value_t = 300)]
    eval_hands: u64,

    #[arg(long, default_value_t = 0)]
    seed: u64,

    #[arg(long, default_value = "value_net.safetensors")]
    output: String,

    /// Load weights from this file instead of starting from scratch -
    /// continue training a checkpoint, or pass `--iterations 0` to just
    /// evaluate it.
    #[arg(long)]
    resume: Option<String>,
}

/// One example: encoded features, and the final score the acting seat got.
type Example = (Vec<f32>, f32);

fn main() {
    let cli = Cli::parse();
    let device = Device::Cpu;
    let config = GameConfig::six_card_golf(2);
    let encoder = Encoder::new(&config);

    let net = match &cli.resume {
        Some(path) => ValueNet::load(encoder.len(), path, &device).expect("loading resume weights should not fail"),
        None => ValueNet::new(encoder.len(), &device).expect("building the value network should not fail"),
    };
    let mut opt = AdamW::new(net.varmap().all_vars(), ParamsAdamW { lr: cli.learning_rate, ..Default::default() })
        .expect("building the optimizer should not fail");
    let mut rng = StdRng::seed_from_u64(cli.seed);

    // Bootstrapping from heuristic play only matters when starting cold - an
    // untrained network's own choices are close to noise. A resumed
    // checkpoint is presumably already better than that.
    let bootstrap_from_heuristic = cli.resume.is_none();

    for iteration in 0..cli.iterations {
        let data_seed = cli.seed.wrapping_add(iteration as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let dataset = if iteration == 0 && bootstrap_from_heuristic {
            println!("iteration 0: bootstrapping from heuristic self-play");
            let mut a = HeuristicAgent;
            let mut b = HeuristicAgent;
            generate_self_play_data(&config, &encoder, &mut a, &mut b, cli.games_per_iteration, data_seed)
        } else {
            let mut a = NeuralAgent::with_exploration(&net, encoder.clone(), cli.exploration);
            let mut b = NeuralAgent::with_exploration(&net, encoder.clone(), cli.exploration);
            generate_self_play_data(&config, &encoder, &mut a, &mut b, cli.games_per_iteration, data_seed)
        };
        println!("iteration {iteration}: {} training examples", dataset.len());

        let mut dataset = dataset;
        for epoch in 0..cli.epochs_per_iteration {
            let loss = train_epoch(&net, &mut opt, &mut dataset, cli.batch_size, &mut rng);
            println!("  epoch {epoch}: mse {loss:.4}");
        }

        report(&config, &net, &encoder, cli.eval_hands, cli.seed);
    }

    if cli.iterations == 0 {
        report(&config, &net, &encoder, cli.eval_hands, cli.seed);
    }

    net.save(&cli.output).expect("saving weights should not fail");
    println!("saved weights to {}", cli.output);
}

/// Evaluates the current network against both baselines and prints the results.
fn report(config: &GameConfig, net: &ValueNet, encoder: &Encoder, hands: u64, seed: u64) {
    let mut neural = NeuralAgent::new(net, encoder.clone());
    let mut heuristic = HeuristicAgent;
    println!("{}", eval_head_to_head(config, &mut neural, &mut heuristic, hands, seed));

    let mut neural = NeuralAgent::new(net, encoder.clone());
    let mut random = RandomAgent;
    println!("{}", eval_head_to_head(config, &mut neural, &mut random, hands, seed));
}

/// Plays `games` self-play hands, alternating which agent sits in seat 0 so
/// neither agent's data is all from one seat, and records every decision
/// point of every hand.
fn generate_self_play_data(
    config: &GameConfig,
    encoder: &Encoder,
    agent_a: &mut dyn Agent,
    agent_b: &mut dyn Agent,
    games: u64,
    seed: u64,
) -> Vec<Example> {
    let mut examples = Vec::new();
    for game in 0..games {
        let deal_seed = seed.wrapping_add(game).wrapping_mul(0x2545_F491_4F6C_DD1D);
        let mut deal_rng = StdRng::seed_from_u64(deal_seed);
        let mut decision_rng = StdRng::seed_from_u64(deal_seed ^ 0xD1CE);

        let agents: [&mut dyn Agent; 2] = if game % 2 == 0 { [agent_a, agent_b] } else { [agent_b, agent_a] };
        record_hand(config, encoder, agents, &mut deal_rng, &mut decision_rng, &mut examples);
    }
    examples
}

fn record_hand(
    config: &GameConfig,
    encoder: &Encoder,
    agents: [&mut dyn Agent; 2],
    deal_rng: &mut StdRng,
    decision_rng: &mut StdRng,
    examples: &mut Vec<Example>,
) {
    let space = golf_sim::ActionSpace::new(config);
    let mut state = GameState::new_with_rng(config.clone(), deal_rng).expect("validated config");
    let mut per_seat_features: Vec<(usize, Vec<f32>)> = Vec::new();

    while let Some(obs) = Observation::for_actor(&state) {
        let seat = obs.seat;
        let normalized = golf_sim::nn::observation_for_value(&obs);
        per_seat_features.push((seat, encoder.encode(&normalized)));

        let index = agents[seat].act(&obs, &space, decision_rng);
        let action = space
            .action_at(index, &obs.turn)
            .unwrap_or_else(|| panic!("{}: index {index} has no action in {:?}", agents[seat].name(), obs.turn));
        state.apply(action).unwrap_or_else(|err| panic!("{}: {err}", agents[seat].name()));
    }

    let scores = state.scores();
    examples.extend(per_seat_features.into_iter().map(|(seat, features)| (features, scores[seat] as f32)));
}

fn train_epoch(net: &ValueNet, opt: &mut AdamW, dataset: &mut [Example], batch_size: usize, rng: &mut StdRng) -> f32 {
    dataset.shuffle(rng);
    let dim = net.input_dim();
    let mut total_loss = 0.0f32;
    let mut batches = 0u32;

    for chunk in dataset.chunks(batch_size) {
        let mut xs = vec![0f32; chunk.len() * dim];
        let mut ys = vec![0f32; chunk.len()];
        for (i, (features, target)) in chunk.iter().enumerate() {
            xs[i * dim..(i + 1) * dim].copy_from_slice(features);
            ys[i] = *target;
        }
        let input = Tensor::from_slice(&xs, (chunk.len(), dim), net.device()).expect("building input batch");
        let target = Tensor::from_slice(&ys, (chunk.len(), 1), net.device()).expect("building target batch");
        let prediction = net.forward(&input).expect("forward pass");
        let loss = candle_nn::loss::mse(&prediction, &target).expect("loss computation");
        opt.backward_step(&loss).expect("optimizer step");
        total_loss += loss.to_vec0::<f32>().expect("loss should be a scalar");
        batches += 1;
    }
    total_loss / batches.max(1) as f32
}
