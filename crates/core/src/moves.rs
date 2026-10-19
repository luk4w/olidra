use std::sync::LazyLock;

use cozy_chess::{Board, Move, Piece, Square};

// indice de lance da policy no estilo Lc0 (1858 saidas), no tabuleiro canonico (quem joga = brancas);
// a ordem precisa ser a mesma usada no treino da policy:
// - 1792 pares (origem, destino) alcançaveis por dama ou cavalo, em ordem crescente de origem e destino;
// - 66 subpromoções: origem na 7a fileira, destino na 8a, peça N, B, R.
// a promoção a dama usa o par comum; o roque (rei captura torre) ja esta entre os pares
pub const N_MOVES: usize = 1858;

struct Table {
    // (promo << 12 | to << 6 | from) -> indice, u16::MAX = fora da tabela; promo 0 normal/dama, 1 N, 2 B, 3 R
    index: Vec<u16>,
}

fn queen_or_knight(from: usize, to: usize) -> bool {
    let df = (to % 8).abs_diff(from % 8);
    let dr = (to / 8).abs_diff(from / 8);
    df == 0 || dr == 0 || df == dr || (df.min(dr) == 1 && df.max(dr) == 2)
}

static TABLE: LazyLock<Table> = LazyLock::new(|| {
    let mut index = vec![u16::MAX; 1 << 14];
    let mut n = 0u16;
    for from in 0..64 {
        for to in 0..64 {
            if from != to && queen_or_knight(from, to) {
                index[to << 6 | from] = n;
                n += 1;
            }
        }
    }
    assert_eq!(n, 1792);
    for from in 48usize..56 {
        for to in [from + 7, from + 8, from + 9] {
            if (to % 8).abs_diff(from % 8) <= 1 {
                for promo in 1..=3 {
                    index[promo << 12 | to << 6 | from] = n;
                    n += 1;
                }
            }
        }
    }
    assert_eq!(n as usize, N_MOVES);
    Table { index }
});

// indice de policy de um lance legal do tabuleiro (representação do cozy-chess: roque = rei captura torre)
pub fn policy_index(board: &Board, mv: Move) -> usize {
    let stm = board.side_to_move();
    let from = mv.from.relative_to(stm) as usize;
    let to = mv.to.relative_to(stm) as usize;
    let promo = match mv.promotion {
        Some(Piece::Knight) => 1,
        Some(Piece::Bishop) => 2,
        Some(Piece::Rook) => 3,
        _ => 0,
    };
    let i = TABLE.index[promo << 12 | to << 6 | from];
    debug_assert!(i != u16::MAX, "lance fora da tabela: {mv}");
    i as usize
}

// lance compacto (u16) para guardar na arvore: from | to << 6 | promo << 12 (promo 0 = nenhuma, 1 N, 2 B, 3 R, 4 Q)
pub fn pack(mv: Move) -> u16 {
    let promo = mv.promotion.map_or(0, |p| p as u16); // Knight = 1 .. Queen = 4
    mv.from as u16 | (mv.to as u16) << 6 | promo << 12
}

pub fn unpack(m: u16) -> Move {
    let promotion = match m >> 12 {
        0 => None,
        p => Some(Piece::index(p as usize)),
    };
    Move {
        from: Square::index((m & 63) as usize),
        to: Square::index((m >> 6 & 63) as usize),
        promotion,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legal_moves(board: &Board) -> Vec<Move> {
        let mut v = Vec::new();
        board.generate_moves(|ms| {
            v.extend(ms);
            false
        });
        v
    }

    // partidas aleatorias: todo lance legal tem indice unico e o pack/unpack volta igual
    #[test]
    fn lances_legais_tem_indice_unico() {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut rand = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..300 {
            let mut board = Board::default();
            for _ in 0..200 {
                let moves = legal_moves(&board);
                if moves.is_empty() {
                    break;
                }
                let mut seen = vec![false; N_MOVES];
                for &mv in &moves {
                    let i = policy_index(&board, mv);
                    assert!(i < N_MOVES && !seen[i], "{mv} em {board}");
                    seen[i] = true;
                    assert_eq!(unpack(pack(mv)), mv);
                }
                board.play_unchecked(moves[rand() as usize % moves.len()]);
            }
        }
    }

    #[test]
    fn roque_e_promocao_nas_duas_cores() {
        // roque curto das brancas e das pretas caem no mesmo indice (e1h1 canonico)
        let w = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", false).unwrap();
        let b = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", false).unwrap();
        assert_eq!(
            policy_index(&w, "e1h1".parse().unwrap()),
            policy_index(&b, "e8h8".parse().unwrap())
        );
        assert_eq!(
            policy_index(&w, "e1a1".parse().unwrap()),
            policy_index(&b, "e8a8".parse().unwrap())
        );
        // promoção: dama no par comum, subpromoções separadas
        let w = Board::from_fen("8/P6k/8/8/8/8/p6K/8 w - - 0 1", false).unwrap();
        let b = Board::from_fen("8/P6k/8/8/8/8/p6K/8 b - - 0 1", false).unwrap();
        let q = policy_index(&w, "a7a8q".parse().unwrap());
        assert_eq!(q, policy_index(&b, "a2a1q".parse().unwrap()));
        let under: Vec<_> = ["a7a8n", "a7a8b", "a7a8r"]
            .iter()
            .map(|m| policy_index(&w, m.parse().unwrap()))
            .collect();
        assert!(under.iter().all(|&i| i >= 1792 && i != q));
    }
}
