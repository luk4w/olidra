// tabuleiro, entradas das redes, indice de lances e avaliador

pub mod eval;
pub mod features;
pub mod moves;
pub mod perft;

pub use cozy_chess;

pub const NAME: &str = "Olidra";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
