//! A small value network: predicts a player's expected final score from an
//! encoded [`crate::repr::Observation`]. Trained by self-play Monte Carlo
//! regression (`src/bin/train_value_net.rs`) and used by `NeuralAgent` for
//! 1-ply expectimax.

use crate::repr::{Observation, TurnView};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};
use std::path::Path;

const HIDDEN: [usize; 2] = [64, 32];

/// The canonical framing a position is valued in: same grids and belief
/// state, but with `turn` reset to `Draw`. `ValueNet` is always queried
/// through this framing - both by `NeuralAgent`'s lookahead (which
/// hypothetically edits a grid, then asks what it's worth going forward) and
/// by training data (built from real game positions in whatever phase they
/// actually occurred in) - so a prediction means the same thing regardless
/// of which real phase a position came from, and training sees the same
/// input distribution inference queries.
pub fn observation_for_value(obs: &Observation) -> Observation {
    let mut next = obs.clone();
    next.turn = TurnView::Draw;
    next
}

/// Two hidden ReLU layers, one scalar output - small enough to train and run
/// on CPU with no batching needed.
pub struct ValueNet {
    l1: Linear,
    l2: Linear,
    l3: Linear,
    varmap: VarMap,
    device: Device,
    input_dim: usize,
}

impl ValueNet {
    pub fn new(input_dim: usize, device: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let l1 = linear(input_dim, HIDDEN[0], vb.pp("l1"))?;
        let l2 = linear(HIDDEN[0], HIDDEN[1], vb.pp("l2"))?;
        let l3 = linear(HIDDEN[1], 1, vb.pp("l3"))?;
        Ok(ValueNet { l1, l2, l3, varmap, device: device.clone(), input_dim })
    }

    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Batched forward pass: `input` is `[batch, input_dim]`, output `[batch, 1]`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.l1.forward(input)?.relu()?;
        let x = self.l2.forward(&x)?.relu()?;
        self.l3.forward(&x)
    }

    /// Predicts a scalar value for one feature vector.
    pub fn predict(&self, features: &[f32]) -> Result<f32> {
        let input = Tensor::from_slice(features, (1, features.len()), &self.device)?;
        let output = self.forward(&input)?;
        Ok(output.to_vec2::<f32>()?[0][0])
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.varmap.save(path)
    }

    /// Builds a fresh network of the given shape, then loads saved weights
    /// into it. `input_dim` must match what the file was saved with.
    pub fn load<P: AsRef<Path>>(input_dim: usize, path: P, device: &Device) -> Result<Self> {
        let mut net = Self::new(input_dim, device)?;
        net.varmap.load(path)?;
        Ok(net)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_returns_a_finite_number() {
        let device = Device::Cpu;
        let net = ValueNet::new(10, &device).unwrap();
        let value = net.predict(&[0.5; 10]).unwrap();
        assert!(value.is_finite());
    }

    #[test]
    fn forward_is_batch_shaped() {
        let device = Device::Cpu;
        let net = ValueNet::new(10, &device).unwrap();
        let input = Tensor::zeros((4, 10), DType::F32, &device).unwrap();
        let output = net.forward(&input).unwrap();
        assert_eq!(output.dims(), &[4, 1]);
    }

    #[test]
    fn save_and_load_round_trips_predictions() {
        let device = Device::Cpu;
        let net = ValueNet::new(10, &device).unwrap();
        let features = [0.1, 0.2, -0.3, 0.4, 0.0, 0.9, -0.5, 0.25, 0.75, -1.0];
        let before = net.predict(&features).unwrap();

        let dir = std::env::temp_dir().join(format!("golf_sim_valuenet_test_{}", std::process::id()));
        let path = dir.join("weights.safetensors");
        std::fs::create_dir_all(&dir).unwrap();
        net.save(&path).unwrap();

        let loaded = ValueNet::load(10, &path, &device).unwrap();
        let after = loaded.predict(&features).unwrap();
        assert_eq!(before, after, "loaded weights must reproduce the same prediction");

        std::fs::remove_dir_all(&dir).ok();
    }
}
