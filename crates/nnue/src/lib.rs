// redes value e policy no estilo NNUE: a posição vira uma lista de linhas ativas (features::RelScan)
// as linhas de uma tabela [linhas][H] em int8 são somadas em int16, passam pela SCReLU e a camada de saída é calculada em float
// As linhas são recalculadas do zero a cada avaliação (sem acumulador incremental)

use std::path::Path;
use std::{fs, io};

use olidra_core::cozy_chess::{Board, Move};
use olidra_core::eval::{CP_SCALE, Evaluator, material_cp, uniform_priors};
use olidra_core::features::{N_REL, RelScan};
use olidra_core::moves::{N_MOVES, policy_index};

// arquivo da rede (little-endian, quantizada pelo training/export.py): magic (FRV8 value, FRP8 policy), u32 H, u32 linhas (N_REL),
// tabela i8[linhas][H], ft_b i16[H], 1/escala f32[H] e o resto em f32 (value: out_w[2H], out_b; policy: out_w[1858][H], out_b[1858])
fn read_rel(
    bytes: &[u8],
    magic: &[u8; 4],
    tail: impl Fn(usize) -> usize,
) -> io::Result<(Table, Vec<f32>)> {
    let bad = |msg: &str| io::Error::new(io::ErrorKind::InvalidData, msg.to_string());
    if bytes.get(..4) != Some(&magic[..]) {
        return Err(bad(&format!(
            "cabeçalho invalido (esperado {})",
            String::from_utf8_lossy(magic)
        )));
    }
    let u32_at = |i: usize| {
        bytes
            .get(i..i + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
            .ok_or_else(|| bad("cabeçalho curto"))
    };
    let (h, n) = (u32_at(4)?, u32_at(8)?);
    if n != N_REL {
        return Err(bad(&format!("{n} linhas, o motor espera {N_REL}")));
    }
    if h > MAX_H {
        return Err(bad(&format!("H {h}: no maximo {MAX_H}")));
    }
    if bytes.len() != 12 + n * h + 2 * h + 4 * h + 4 * tail(h) {
        return Err(bad("tamanho não bate com a arquitetura"));
    }
    let (ft_w, rest) = bytes[12..].split_at(n * h);
    let (ft_b, rest) = rest.split_at(2 * h);
    let f32s = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    let (inv, rest) = rest.split_at(4 * h);
    let table = Table {
        hidden: h,
        ft_w: ft_w.iter().map(|&b| b as i8).collect(),
        ft_b: ft_b
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        inv: f32s(inv),
    };
    Ok((table, f32s(rest)))
}

fn read_file(path: &Path) -> io::Result<Vec<u8>> {
    fs::read(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))
}

// produto escalar com 8 somas parciais independentes: o compilador vetoriza (AVX2);
// uma soma unica em sequencia não vetoriza, porque mudar a ordem muda o arredondamento
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ca
        .remainder()
        .iter()
        .zip(cb.remainder())
        .map(|(&x, &y)| x * y)
        .sum();
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    acc.iter().sum::<f32>() + tail
}

// a tabela e o bias são quantizados em int8 / int16 com uma escala por coluna (neurônio); é onde está o custo (somar as linhas).
// A camada de saída continua em float: com os pesos de saída arredondados, os erros fixos de cada peso se somam sempre na mesma direção
// e a value fica enviesada (cerca de +0,007 na escala [-1, 1])
const MAX_H: usize = 1024;

struct Table {
    hidden: usize,
    ft_w: Vec<i8>,
    ft_b: Vec<i16>,
    inv: Vec<f32>, // 1 / escala de cada coluna
}

impl Table {
    // soma as linhas ativas sobre o bias e aplica a SCReLU (na escala 1)
    fn hidden(&self, feats: &[u16], out: &mut [f32]) {
        let h = self.hidden;
        let mut acc = [0i16; MAX_H];
        let acc = &mut acc[..h];
        acc.copy_from_slice(&self.ft_b);
        for &f in feats {
            let col = &self.ft_w[f as usize * h..][..h];
            for (a, &c) in acc.iter_mut().zip(col) {
                *a = a.wrapping_add(c as i16);
            }
        }
        for ((o, &a), &k) in out.iter_mut().zip(acc.iter()).zip(&self.inv) {
            let c = (a as f32 * k).clamp(0.0, 1.0);
            *o = c * c;
        }
    }
}

// (linhas -> H) x 2 perspectivas -> 1, SCReLU
pub struct ValueNet {
    table: Table,
    out_w: Vec<f32>,
    out_b: f32,
}

impl ValueNet {
    pub fn load(path: &Path) -> io::Result<Self> {
        Self::from_bytes(&read_file(path)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let (table, rest) = read_rel(bytes, b"FRV8", |h| 2 * h + 1)?;
        let h = table.hidden;
        Ok(Self {
            table,
            out_w: rest[..2 * h].to_vec(),
            out_b: rest[2 * h],
        })
    }

    // valor em [-1, 1] na visão de quem joga
    pub fn evaluate(&self, board: &Board) -> f32 {
        let h = self.table.hidden;
        let stm = board.side_to_move();
        let scan = RelScan::new(board);
        let mut feats = Vec::with_capacity(128);
        let mut hid = vec![0.0; h];
        let mut sum = self.out_b;
        for (k, p) in [stm, !stm].into_iter().enumerate() {
            scan.encode(board, p, &mut feats);
            self.table.hidden(&feats, &mut hid);
            sum += dot(&hid, &self.out_w[k * h..][..h]);
        }
        sum.tanh()
    }
}

// linhas -> H -> 1858, SCReLU; saida calculada so para os lances legais
pub struct PolicyNet {
    table: Table,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
}

impl PolicyNet {
    pub fn load(path: &Path) -> io::Result<Self> {
        Self::from_bytes(&read_file(path)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let (table, rest) = read_rel(bytes, b"FRP8", |h| N_MOVES * h + N_MOVES)?;
        let (out_w, out_b) = rest.split_at(N_MOVES * table.hidden);
        Ok(Self {
            table,
            out_w: out_w.to_vec(),
            out_b: out_b.to_vec(),
        })
    }

    // softmax dos logits dos lances legais
    pub fn priors(&self, board: &Board, moves: &[Move], out: &mut [f32]) {
        let h = self.table.hidden;
        let mut feats = Vec::with_capacity(128);
        RelScan::new(board).encode(board, board.side_to_move(), &mut feats);
        let mut hid = vec![0.0; h];
        self.table.hidden(&feats, &mut hid);
        for (o, &mv) in out.iter_mut().zip(moves) {
            let i = policy_index(board, mv);
            *o = self.out_b[i] + dot(&hid, &self.out_w[i * h..][..h]);
        }
        softmax(out);
    }
}

fn softmax(out: &mut [f32]) {
    let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for o in out.iter_mut() {
        *o = (*o - max).exp();
        total += *o;
    }
    out.iter_mut().for_each(|o| *o /= total);
}

// redes reais; sem value usa material, sem policy usa prior uniforme
pub struct NnueEvaluator {
    pub value: Option<ValueNet>,
    pub policy: Option<PolicyNet>,
}

impl Evaluator for NnueEvaluator {
    fn value(&self, board: &Board) -> f32 {
        match &self.value {
            Some(v) => v.evaluate(board),
            None => (material_cp(board) as f32 / CP_SCALE).tanh(),
        }
    }

    fn priors(&self, board: &Board, moves: &[Move], priors: &mut [f32]) {
        match &self.policy {
            Some(p) => p.priors(board, moves, priors),
            None => uniform_priors(priors),
        }
    }
}
