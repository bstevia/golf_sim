use super::card::{Card, Rank, Suit};
use rand::rng;
use rand::seq::SliceRandom;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Deck {
    cards: Vec<Card>,
}

impl Deck {
    pub fn new(num_decks: usize) -> Self {
        let mut cards = Vec::with_capacity(52 * num_decks);
        for _ in 0..num_decks {
            for suit in [Suit::Hearts, Suit::Diamonds, Suit::Clubs, Suit::Spades] {
                for rank in Rank::ALL {
                    cards.push(Card::new(rank, suit));
                }
            }
        }
        Deck { cards }
    }

    pub fn shuffle(&mut self) {
        self.cards.shuffle(&mut rng());
    }

    pub fn shuffle_with<R: Rng>(&mut self, rng: &mut R) {
        self.cards.shuffle(rng);
    }

    pub fn deal(&mut self) -> Option<Card> {
        self.cards.pop()
    }

    pub fn remaining(&self) -> usize {
        self.cards.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }

    /// Takes an explicit RNG so a reshuffle stays reproducible under a seed.
    pub fn refill_and_shuffle<R: Rng>(&mut self, cards: Vec<Card>, rng: &mut R) {
        self.cards = cards;
        self.shuffle_with(rng);
    }
}
