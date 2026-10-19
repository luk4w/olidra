use cozy_chess::{Board, Color, Piece};

// peça x casa: 12 tipos (6 meus, 6 do oponente) x 64 casas, na perspectiva de quem é codificado
pub const N_FEATURES: usize = 768;

fn piece_attacks(
    t: Piece,
    sq: cozy_chess::Square,
    c: Color,
    occupied: cozy_chess::BitBoard,
) -> cozy_chess::BitBoard {
    match t {
        Piece::Pawn => cozy_chess::get_pawn_attacks(sq, c),
        Piece::Knight => cozy_chess::get_knight_moves(sq),
        Piece::Bishop => cozy_chess::get_bishop_moves(sq, occupied),
        Piece::Rook => cozy_chess::get_rook_moves(sq, occupied),
        Piece::Queen => {
            cozy_chess::get_bishop_moves(sq, occupied) | cozy_chess::get_rook_moves(sq, occupied)
        }
        Piece::King => cozy_chess::get_king_moves(sq),
    }
}

// entradas das redes ("relações" entre as peças): cada posição vira uma lista de linhas ativas, na perspectiva p
// (casas espelhadas quando p é das pretas); a codificação precisa ser a mesma usada no treino:
//   base peça x casa: codigo * 64 + casa (codigo 0-5 de p, 6-11 do oponente);
//   ataque: 768 + ((atacante * 12 + alvo) * 64 + casa do alvo) * 2 + alvo defendido, uma linha por peça que ataca
//     uma peça da outra cor;
//   vazias atacadas por p: REL_OFF_V + casa * 2 + (0 so p ataca, 1 disputada);
//   casas de xeque (vazias ou com peça do oponente): REL_OFF_X + casa * 32 + mascara dos tipos de p que vão à casa dando
//     xeque direto (1 P, 2 N, 4 B, 8 R, 16 Q).
pub const REL_OFF_V: usize = 768 + 12 * 12 * 64 * 2;
pub const REL_OFF_X: usize = REL_OFF_V + 128;
pub const N_REL: usize = REL_OFF_X + 64 * 32;

// ataques de cada peça, calculados uma vez e codificados em qualquer perspectiva (a value usa as duas, a policy uma)
pub struct RelScan {
    att: [cozy_chess::BitBoard; 64],
    by: [cozy_chess::BitBoard; 2],
    by_type: [[cozy_chess::BitBoard; 6]; 2],
}

impl RelScan {
    pub fn new(board: &Board) -> Self {
        let occ = board.occupied();
        let empty = cozy_chess::BitBoard::EMPTY;
        let (mut att, mut by, mut by_type) = ([empty; 64], [empty; 2], [[empty; 6]; 2]);
        for sq in occ {
            let (t, c) = (board.piece_on(sq).unwrap(), board.color_on(sq).unwrap());
            let a = piece_attacks(t, sq, c, occ);
            att[sq as usize] = a;
            by[c as usize] |= a;
            by_type[c as usize][t as usize] |= a;
        }
        Self { att, by, by_type }
    }

    pub fn encode(&self, board: &Board, p: Color, out: &mut Vec<u16>) {
        out.clear();
        let occ = board.occupied();
        let code = |c: Color, t: Piece| (c != p) as usize * 6 + t as usize;
        for x in occ {
            let (tx, cx) = (board.piece_on(x).unwrap(), board.color_on(x).unwrap());
            out.push((code(cx, tx) * 64 + x.relative_to(p) as usize) as u16);
            for y in self.att[x as usize] & board.colors(!cx) {
                let (ty, cy) = (board.piece_on(y).unwrap(), board.color_on(y).unwrap());
                let defended = self.by[cy as usize].has(y) as usize;
                out.push(
                    (N_FEATURES
                        + ((code(cx, tx) * 12 + code(cy, ty)) * 64 + y.relative_to(p) as usize) * 2
                        + defended) as u16,
                );
            }
        }
        let (me, him) = (self.by[p as usize], self.by[!p as usize]);
        for sq in !occ & me {
            out.push((REL_OFF_V + sq.relative_to(p) as usize * 2 + him.has(sq) as usize) as u16);
        }
        // casas de xeque: de onde cada tipo de p atacaria o rei do oponente (raios a partir do rei, tabuleiro atual)
        let ek = board.king(!p);
        let pawn_from = cozy_chess::get_pawn_attacks(ek, !p);
        let knight_from = cozy_chess::get_knight_moves(ek);
        let diag_from = cozy_chess::get_bishop_moves(ek, occ);
        let orth_from = cozy_chess::get_rook_moves(ek, occ);
        let mine = &self.by_type[p as usize];
        let my_pawns = board.colored_pieces(p, Piece::Pawn);
        let fwd: i32 = if p == Color::White { 8 } else { -8 };
        for s in (!occ | board.colors(!p)) & (pawn_from | knight_from | diag_from | orth_from) {
            let mut mask = 0;
            if pawn_from.has(s) {
                let rel_rank = s.relative_to(p) as usize / 8;
                let at = |d: i32| cozy_chess::Square::index((s as i32 - d) as usize);
                let reaches = if occ.has(s) {
                    mine[Piece::Pawn as usize].has(s)
                } else {
                    rel_rank >= 1
                        && (my_pawns.has(at(fwd))
                            || (rel_rank == 3 && !occ.has(at(fwd)) && my_pawns.has(at(2 * fwd))))
                };
                mask |= reaches as usize;
            }
            mask |= ((knight_from.has(s) && mine[Piece::Knight as usize].has(s)) as usize) << 1;
            mask |= ((diag_from.has(s) && mine[Piece::Bishop as usize].has(s)) as usize) << 2;
            mask |= ((orth_from.has(s) && mine[Piece::Rook as usize].has(s)) as usize) << 3;
            mask |= (((diag_from | orth_from).has(s) && mine[Piece::Queen as usize].has(s))
                as usize)
                << 4;
            if mask != 0 {
                out.push((REL_OFF_X + s.relative_to(p) as usize * 32 + mask) as u16);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posicao_espelhada_tem_as_mesmas_linhas() {
        // mesma posição com as cores trocadas e o tabuleiro espelhado, nas duas perspectivas
        let a = Board::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            false,
        )
        .unwrap();
        let b = Board::from_fen(
            "r3k2r/pppbbppp/2n2q1P/1P2p3/3pn3/BN2PNP1/P1PPQPB1/R3K2R b KQkq - 0 1",
            false,
        )
        .unwrap();
        let (sa, sb) = (RelScan::new(&a), RelScan::new(&b));
        let (mut fa, mut fb) = (Vec::new(), Vec::new());
        for (pa, pb) in [
            (a.side_to_move(), b.side_to_move()),
            (!a.side_to_move(), !b.side_to_move()),
        ] {
            sa.encode(&a, pa, &mut fa);
            sb.encode(&b, pb, &mut fb);
            fa.sort();
            fb.sort();
            assert_eq!(fa, fb);
            assert!(fa.iter().all(|&i| (i as usize) < N_REL));
        }
    }
}
