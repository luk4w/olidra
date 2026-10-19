use cozy_chess::Board;

// conta as folhas da arvore de lances legais ate a profundidade pedida
pub fn perft(board: &Board, depth: u8) -> u64 {
    let mut nodes = 0;
    match depth {
        0 => nodes = 1,
        // no penultimo nivel basta somar quantos lances existem
        1 => {
            board.generate_moves(|moves| {
                nodes += moves.len() as u64;
                false
            });
        }
        _ => {
            board.generate_moves(|moves| {
                for mv in moves {
                    let mut child = board.clone();
                    child.play_unchecked(mv);
                    nodes += perft(&child, depth - 1);
                }
                false
            });
        }
    }
    nodes
}

// posições de referencia do chessprogramming wiki: (nome, FEN, profundidade, nos)
pub const REFERENCE: [(&str, &str, u8, u64); 6] = [
    (
        "inicial",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        6,
        119_060_324,
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        5,
        193_690_690,
    ),
    (
        "posicao 3",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        6,
        11_030_083,
    ),
    (
        "posicao 4",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        5,
        15_833_292,
    ),
    (
        "posicao 5",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        5,
        89_941_194,
    ),
    (
        "posicao 6",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        5,
        164_075_551,
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    // versao rapida: uma profundidade a menos, roda no cargo test comum
    #[test]
    fn perft_raso_bate_com_referencia() {
        // nos esperados com uma profundidade a menos que a da tabela REFERENCE
        let rasos = [
            4_865_609_u64,
            4_085_603,
            674_624,
            422_333,
            2_103_487,
            3_894_594,
        ];
        for ((nome, fen, depth, _), nodes) in REFERENCE.iter().zip(rasos) {
            let board = Board::from_fen(fen, false).unwrap();
            assert_eq!(perft(&board, depth - 1), nodes, "{nome}");
        }
    }

    // versao completa: cargo test --release -- --ignored
    #[test]
    #[ignore]
    fn perft_completo_bate_com_referencia() {
        for (nome, fen, depth, nodes) in REFERENCE {
            let board = Board::from_fen(fen, false).unwrap();
            assert_eq!(perft(&board, depth), nodes, "{nome}");
        }
    }
}
