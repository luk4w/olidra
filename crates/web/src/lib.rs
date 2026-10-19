// interface do motor para o navegador (wasm-bindgen): carrega as redes a partir dos bytes e busca uma posição dada como na UCI

use std::sync::atomic::AtomicBool;
use std::time::Duration;

use olidra_core::cozy_chess::Board;
use olidra_core::cozy_chess::util::{display_uci_move, parse_uci_move};
use olidra_nnue::{NnueEvaluator, PolicyNet, ValueNet};
use olidra_search::{Limits, Params, Search, Tree};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Engine {
    eval: NnueEvaluator,
    tree: Tree,
}

#[wasm_bindgen]
pub struct Reply {
    best: String,
    score_cp: i32,
    nodes: u32,
    second: String,
    second_cp: i32,
    has_second: bool,
    seq: String,
    root: String,
}

#[wasm_bindgen]
impl Reply {
    #[wasm_bindgen(getter)]
    pub fn best(&self) -> String {
        self.best.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn score_cp(&self) -> i32 {
        self.score_cp
    }

    #[wasm_bindgen(getter)]
    pub fn nodes(&self) -> u32 {
        self.nodes
    }

    #[wasm_bindgen(getter)]
    pub fn second(&self) -> Option<String> {
        self.has_second.then(|| self.second.clone())
    }

    #[wasm_bindgen(getter)]
    pub fn second_cp(&self) -> Option<i32> {
        self.has_second.then_some(self.second_cp)
    }

    #[wasm_bindgen(getter)]
    pub fn seq(&self) -> String {
        self.seq.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn root(&self) -> String {
        self.root.clone()
    }
}

// fatia da análise sem fim: as linhas da raiz (MultiPV) com a nota e a variante principal
#[wasm_bindgen]
pub struct Live {
    nodes: u32,
    depth: u32,
    seldepth: u32,
    hashfull: f32,
    lines: String,
}

#[wasm_bindgen]
impl Live {
    #[wasm_bindgen(getter)]
    pub fn nodes(&self) -> u32 {
        self.nodes
    }

    #[wasm_bindgen(getter)]
    pub fn depth(&self) -> u32 {
        self.depth
    }

    #[wasm_bindgen(getter)]
    pub fn seldepth(&self) -> u32 {
        self.seldepth
    }

    #[wasm_bindgen(getter)]
    pub fn hashfull(&self) -> f32 {
        self.hashfull
    }

    // "cp:visitas:lance lance ...;" uma linha por lance, do mais visitado para o menos
    #[wasm_bindgen(getter)]
    pub fn lines(&self) -> String {
        self.lines.clone()
    }
}

fn js_err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

// posição a partir da fen e dos lances em UCI, com os hashes da partida (para a repetição)
fn position(fen: &str, moves: &str) -> Result<(Board, Vec<u64>), JsError> {
    let mut board = Board::from_fen(fen, false).map_err(|e| js_err(format!("{e:?}")))?;
    let mut history = vec![board.hash()];
    for m in moves.split_whitespace() {
        let mv = parse_uci_move(&board, m).map_err(|_| js_err(format!("lance ilegal: {m}")))?;
        if !board.is_legal(mv) {
            return Err(js_err(format!("lance ilegal: {m}")));
        }
        board.play_unchecked(mv);
        history.push(board.hash());
    }
    Ok((board, history))
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new(value: &[u8], policy: &[u8], hash_mb: usize) -> Result<Engine, JsError> {
        Ok(Engine {
            eval: NnueEvaluator {
                value: Some(ValueNet::from_bytes(value).map_err(js_err)?),
                policy: Some(PolicyNet::from_bytes(policy).map_err(js_err)?),
            },
            tree: Tree::new(hash_mb << 20),
        })
    }

    // análise sem fim em fatias de `ms` milissegundos: `fresh` começa uma árvore nova; senão continua a da fatia
    // anterior, que tem de ser da mesma posição
    pub fn ponder(&self, fen: &str, moves: &str, ms: u32, multipv: u32, fresh: bool) -> Result<Live, JsError> {
        let (board, history) = position(fen, moves)?;
        let params = Params {
            multipv: multipv.max(2) as usize,
            ..Params::default()
        };
        let search = if fresh {
            Search::new(&self.tree, board, history, &self.eval, params)
        } else {
            Search::resume(&self.tree, board, history, &self.eval, params)
        };
        let limits = Limits {
            time: Some(Duration::from_millis(ms.max(1) as u64)),
            ..Limits::default()
        };
        let info = search.run(&limits, &AtomicBool::new(false), |_| {}).info;
        let lines = if info.lines.is_empty() {
            format!("{}:{}:{}", info.score_cp, info.nodes, info.pv.join(" "))
        } else {
            info.lines
                .iter()
                .map(|l| format!("{}:{}:{}", l.score_cp, l.visits, l.pv.join(" ")))
                .collect::<Vec<_>>()
                .join(";")
        };
        Ok(Live {
            nodes: info.nodes as u32,
            depth: info.depth as u32,
            seldepth: info.seldepth as u32,
            hashfull: info.hashfull,
            lines,
        })
    }

    // fen inicial e lances em UCI separados por espaço; para no que vier primeiro, nós ou milissegundos (0 = sem limite)
    pub fn think(&self, fen: &str, moves: &str, nodes: u32, ms: u32) -> Result<Reply, JsError> {
        let (board, history) = position(fen, moves)?;
        let limits = Limits {
            nodes: (nodes > 0).then_some(nodes as u64),
            time: (ms > 0).then(|| Duration::from_millis(ms as u64)),
            depth: None,
        };
        let params = Params {
            multipv: 2,
            ..Params::default()
        };
        let search = Search::new(&self.tree, board.clone(), history, &self.eval, params);
        let result = search.run(&limits, &AtomicBool::new(false), |_| {});
        let second_line = result.info.lines.get(1).filter(|l| l.visits > 0);
        let min_visits = (result.info.nodes / 256).clamp(8, 256) as u32;
        let seq = search
            .sequence(min_visits)
            .iter()
            .map(|st| {
                let nat = st.natural_cp.map_or("-".into(), |c| c.to_string());
                format!(
                    "{}:{}:{:.2}:{:.2}:{:.3}:{:.3}:{}:{}:{nat}",
                    st.mv, st.visits, st.d, st.h, st.prior, st.share, st.score_cp, st.natural_mv
                )
            })
            .collect::<Vec<_>>()
            .join(";");
        let prof = search.instrumental_profile();
        let root = prof.split(';').next().unwrap_or("-").to_string();
        Ok(Reply {
            best: display_uci_move(&board, result.best).to_string(),
            score_cp: result.info.score_cp,
            nodes: result.info.nodes as u32,
            second: second_line
                .and_then(|l| l.pv.first().cloned())
                .unwrap_or_default(),
            second_cp: second_line.map_or(0, |l| l.score_cp),
            has_second: second_line.is_some(),
            seq,
            root,
        })
    }
}
