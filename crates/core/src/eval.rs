use cozy_chess::{Board, Color, Move, Piece};

// avaliação de uma folha, em duas partes (a busca pode pedir so o valor na primeira visita):
// valor em [-1, 1] na visão de quem joga, e um prior por lance legal
pub trait Evaluator: Send + Sync {
    fn value(&self, board: &Board) -> f32;
    fn priors(&self, board: &Board, moves: &[Move], priors: &mut [f32]);
}

pub const CP_SCALE: f32 = 600.0;

pub fn uniform_priors(priors: &mut [f32]) {
    let p = 1.0 / priors.len() as f32;
    priors.fill(p);
}

// material em centipawns na visão de quem joga
pub fn material_cp(board: &Board) -> i32 {
    const VALUES: [i32; 6] = [100, 300, 300, 500, 900, 0];
    let stm = board.side_to_move();
    Piece::ALL
        .iter()
        .map(|&p| {
            VALUES[p as usize]
                * (board.colored_pieces(stm, p).len() as i32
                    - board.colored_pieces(!stm, p).len() as i32)
        })
        .sum()
}

// prior uniforme, valor 0 (testes da busca)
pub struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn value(&self, _: &Board) -> f32 {
        0.0
    }

    fn priors(&self, _: &Board, _: &[Move], priors: &mut [f32]) {
        uniform_priors(priors);
    }
}

// tanh(material / 600), prior uniforme
pub struct MaterialEvaluator;

impl Evaluator for MaterialEvaluator {
    fn value(&self, board: &Board) -> f32 {
        (material_cp(board) as f32 / CP_SCALE).tanh()
    }

    fn priors(&self, _: &Board, _: &[Move], priors: &mut [f32]) {
        uniform_priors(priors);
    }
}

// sem material para dar mate: so reis, ou rei + uma peça menor contra rei
pub fn insufficient_material(board: &Board) -> bool {
    let heavy = board.pieces(Piece::Pawn) | board.pieces(Piece::Rook) | board.pieces(Piece::Queen);
    if !heavy.is_empty() {
        return false;
    }
    let minors = board.pieces(Piece::Knight) | board.pieces(Piece::Bishop);
    minors.len() <= 1
        && [Color::White, Color::Black]
            .iter()
            .all(|&c| (board.colors(c) & minors).len() <= 1)
}
