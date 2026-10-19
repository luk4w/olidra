// MCTS PUCT com várias threads descendo a mesma árvore ao mesmo tempo (virtual loss para espalhá-las).
// A árvore vive em duas arenas pré-alocadas, com campos atômicos:
// - arestas (8 bytes): lance, prior e índice do filho; criadas juntas quando o pai é expandido;
// - nós (32 bytes): visitas e soma dos valores; criados só na primeira visita da aresta (a maioria dos
//   filhos nunca é visitada).
// Quando a memória acaba, a árvore é compactada (fica a parte mais visitada) e a busca continua.

use std::sync::atomic::{
    AtomicBool, AtomicI64, AtomicU8, AtomicU16, AtomicU32, AtomicU64, AtomicUsize, Ordering::*,
};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
// no navegador, std::time::Instant não existe
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

use olidra_core::cozy_chess::util::display_uci_move;
use olidra_core::cozy_chess::{Board, Move, Square};
use olidra_core::eval::{CP_SCALE, Evaluator, insufficient_material};
use olidra_core::moves::{pack, unpack};

const UNEXPANDED: u8 = 0;
const EXPANDING: u8 = 1;
const EXPANDED: u8 = 2;
const TERMINAL: u8 = 3;
// sem espaço para as arestas: o nó vira folha permanente com o valor da primeira avaliação
const FULL: u8 = 4;
// expansão adiada: avaliado so com a value na primeira visita; arestas e policy na segunda
const VISITED: u8 = 5;

// W em ponto fixo: valor x 2^16
const W_SCALE: f64 = 65536.0;
// prior em ponto fixo u16
const P_SCALE: f32 = 65535.0;
// arestas reservadas por nó ao dividir a memória: com a expansão adiada só ~30% dos nós ganham arestas
// (na prática, ~12 arestas por nó criado)
const EDGES_PER_NODE: usize = 16;

#[derive(Default)]
struct Node {
    w: AtomicI64, // soma dos valores na visão de quem joga no nó
    n: AtomicU32,
    vl: AtomicU32,         // descidas em andamento (virtual loss)
    v: AtomicU32,          // bits do f32: valor fixo de nó terminal ou cheio
    first_edge: AtomicU32, // valido depois de state = EXPANDED (Acquire)
    num_edges: AtomicU8,
    state: AtomicU8,
}

impl Node {
    fn reset(&self) {
        self.w.store(0, Relaxed);
        self.n.store(0, Relaxed);
        self.vl.store(0, Relaxed);
        self.v.store(0, Relaxed);
        self.first_edge.store(0, Relaxed);
        self.num_edges.store(0, Relaxed);
        self.state.store(UNEXPANDED, Relaxed);
    }

    fn edges(&self) -> std::ops::Range<usize> {
        let f = self.first_edge.load(Relaxed) as usize;
        f..f + self.num_edges.load(Relaxed) as usize
    }

    fn fixed_value(&self) -> f32 {
        f32::from_bits(self.v.load(Relaxed))
    }

    fn copy_from(&self, o: &Node) {
        self.w.store(o.w.load(Relaxed), Relaxed);
        self.n.store(o.n.load(Relaxed), Relaxed);
        self.vl.store(o.vl.load(Relaxed), Relaxed);
        self.v.store(o.v.load(Relaxed), Relaxed);
        self.first_edge.store(o.first_edge.load(Relaxed), Relaxed);
        self.num_edges.store(o.num_edges.load(Relaxed), Relaxed);
        self.state.store(o.state.load(Relaxed), Relaxed);
    }
}

#[derive(Default)]
struct Edge {
    child: AtomicU32, // 0 = filho ainda não criado (o nó 0 é a raiz, nunca é filho)
    mv: AtomicU16,
    p: AtomicU16,
}

impl Edge {
    fn prior(&self) -> f32 {
        self.p.load(Relaxed) as f32 / P_SCALE
    }

    fn copy_from(&self, o: &Edge) {
        self.child.store(o.child.load(Relaxed), Relaxed);
        self.mv.store(o.mv.load(Relaxed), Relaxed);
        self.p.store(o.p.load(Relaxed), Relaxed);
    }
}

// estatisticas de uma aresta, lidas do filho se ele existir
#[derive(Clone, Copy, Default)]
struct Stats {
    n: u32,
    vl: u32,
    w: f64,
}

impl Stats {
    // valor medio na visão de quem escolhe a aresta
    fn q_parent(&self) -> f32 {
        if self.n == 0 {
            0.0
        } else {
            (-self.w / self.n as f64) as f32
        }
    }
}

// arenas fixas, reaproveitadas entre buscas; so os cursores voltam para o inicio.
// Reserva (1/3 da memoria, metade da arena principal): destino da compactação quando a arena enche
pub struct Tree {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    node_cursor: AtomicUsize,
    edge_cursor: AtomicUsize,
    spare: std::sync::Mutex<(Vec<Node>, Vec<Edge>)>,
}

fn bump(cursor: &AtomicUsize, n: usize, capacity: usize) -> Option<usize> {
    if cursor.load(Relaxed) + n > capacity {
        return None;
    }
    let first = cursor.fetch_add(n, Relaxed);
    (first + n <= capacity).then_some(first)
}

impl Tree {
    // divide a memoria entre nós e arestas
    pub fn new(bytes: usize) -> Self {
        let edges_per_node = EDGES_PER_NODE;
        let unit = size_of::<Node>() + edges_per_node * size_of::<Edge>();
        let total = bytes / unit;
        let nodes = (total - total / 3).clamp(64, u32::MAX as usize / edges_per_node);
        let spare = nodes / 2;
        Self {
            nodes: (0..nodes).map(|_| Node::default()).collect(),
            edges: (0..nodes * edges_per_node)
                .map(|_| Edge::default())
                .collect(),
            node_cursor: AtomicUsize::new(1),
            edge_cursor: AtomicUsize::new(0),
            spare: std::sync::Mutex::new((
                (0..spare).map(|_| Node::default()).collect(),
                (0..spare * edges_per_node)
                    .map(|_| Edge::default())
                    .collect(),
            )),
        }
    }

    pub fn bytes(&self) -> usize {
        let spare = self.spare.lock().unwrap();
        (self.nodes.len() + spare.0.len()) * size_of::<Node>()
            + (self.edges.len() + spare.1.len()) * size_of::<Edge>()
    }

    // falta pouco para encher (folga para as simulações em andamento); arenas pequenas demais não compactam
    fn near_full(&self, threads: usize) -> bool {
        let (mn, me) = (4 * threads + 8, 256 * threads + 512);
        if self.nodes.len() < 16 * mn || self.edges.len() < 16 * me {
            return false;
        }
        self.node_cursor.load(Relaxed) + mn > self.nodes.len()
            || self.edge_cursor.load(Relaxed) + me > self.edges.len()
    }

    // compactação (com a busca parada): fica a subárvore dos nós expandidos com N >= k, com o menor k que cabe na
    // reserva (um nó nunca tem mais visitas que o pai, então o que fica é conexo a partir da raiz).
    // Filhos de nós que ficam e têm N < k viram tocos: guardam N, W e o valor, perdem as arestas e são expandidos de novo se a busca
    // voltar (estado VISITED); nós FULL também voltam a poder expandir. Devolve o k.
    fn compact(&self) -> u32 {
        let spare = self.spare.lock().unwrap();
        let (sn, se) = (&spare.0, &spare.1);
        // (N, arestas, filhos criados) de cada nó expandido alcançável
        let mut items = Vec::new();
        let mut stack = vec![0usize];
        while let Some(i) = stack.pop() {
            let node = &self.nodes[i];
            if node.state.load(Relaxed) != EXPANDED {
                continue;
            }
            let mut kids = 0;
            for e in node.edges() {
                let c = self.edges[e].child.load(Relaxed) as usize;
                if c != 0 {
                    kids += 1;
                    stack.push(c);
                }
            }
            items.push((
                node.n.load(Relaxed),
                node.num_edges.load(Relaxed) as usize,
                kids,
            ));
        }
        items.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        let (mut used_n, mut used_e, mut k) = (1usize, 0usize, 0u32);
        for &(n, e, c) in &items {
            if used_n + c > sn.len() || used_e + e > se.len() {
                k = n + 1;
                break;
            }
            used_n += c;
            used_e += e;
        }
        // a raiz fica sempre expandida
        let k = k.min(self.nodes[0].n.load(Relaxed));

        sn[0].copy_from(&self.nodes[0]);
        let (mut next_n, mut next_e) = (1usize, 0usize);
        let mut queue = vec![(0usize, 0usize)];
        while let Some((old, new)) = queue.pop() {
            let range = self.nodes[old].edges();
            let first = next_e;
            next_e += range.len();
            sn[new].first_edge.store(first as u32, Relaxed);
            for (j, e) in range.enumerate() {
                let dst = &se[first + j];
                dst.copy_from(&self.edges[e]);
                let c = self.edges[e].child.load(Relaxed) as usize;
                if c == 0 {
                    continue;
                }
                let nc = next_n;
                next_n += 1;
                dst.child.store(nc as u32, Relaxed);
                let (src, cd) = (&self.nodes[c], &sn[nc]);
                cd.copy_from(src);
                match src.state.load(Relaxed) {
                    EXPANDED if src.n.load(Relaxed) >= k => queue.push((c, nc)),
                    EXPANDED | FULL => {
                        cd.state.store(VISITED, Relaxed);
                        cd.first_edge.store(0, Relaxed);
                        cd.num_edges.store(0, Relaxed);
                    }
                    _ => {}
                }
            }
        }
        for i in 0..next_n {
            self.nodes[i].copy_from(&sn[i]);
        }
        for e in 0..next_e {
            self.edges[e].copy_from(&se[e]);
        }
        self.node_cursor.store(next_n, Release);
        self.edge_cursor.store(next_e, Release);
        k
    }

    // (nós usados, arestas usadas)
    pub fn usage(&self) -> (usize, usize) {
        (
            self.node_cursor.load(Relaxed).min(self.nodes.len()),
            self.edge_cursor.load(Relaxed).min(self.edges.len()),
        )
    }

    fn reset(&self) {
        self.nodes[0].reset();
        self.node_cursor.store(1, Relaxed);
        self.edge_cursor.store(0, Relaxed);
    }

    fn stats(&self, e: usize) -> Stats {
        match self.edges[e].child.load(Acquire) as usize {
            0 => Stats::default(),
            c => {
                let node = &self.nodes[c];
                Stats {
                    n: node.n.load(Relaxed),
                    vl: node.vl.load(Relaxed),
                    w: node.w.load(Relaxed) as f64 / W_SCALE,
                }
            }
        }
    }

    // filho da aresta, criando o nó na primeira visita; None se a arena de nós encheu
    fn child(&self, e: usize) -> Option<usize> {
        let edge = &self.edges[e];
        let c = edge.child.load(Acquire) as usize;
        if c != 0 {
            return Some(c);
        }
        let k = bump(&self.node_cursor, 1, self.nodes.len())?;
        self.nodes[k].reset();
        // outra thread pode ter criado o filho ao mesmo tempo: fica o dela (o nó reservado aqui se perde)
        match edge.child.compare_exchange(0, k as u32, AcqRel, Acquire) {
            Ok(_) => Some(k),
            Err(other) => Some(other as usize),
        }
    }
}

pub struct Params {
    // c_puct crescente com as visitas do pai (mesma forma do Lc0): cpuct + factor * ln((N + base) / base);
    // factor 0 = c_puct fixo
    pub cpuct: f32,
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    pub threads: usize,
    // linhas no info (MultiPV): os lances mais visitados da raiz, cada um com a sua PV
    pub multipv: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            cpuct: 1.75,
            cpuct_base: 38739.0,
            cpuct_factor: 3.89,
            threads: 1,
            multipv: 1,
        }
    }
}

// nós buscados quando a raiz tem um lance legal apenas
const SINGLE_NODES: u64 = 2048;

#[derive(Default)]
pub struct Limits {
    pub nodes: Option<u64>,
    pub time: Option<Duration>,
    // profundidade = tamanho da variante principal (o `depth` do info)
    pub depth: Option<usize>,
}

pub struct Info {
    pub depth: usize,
    pub seldepth: usize,
    pub nodes: u64,
    pub elapsed: Duration,
    pub score_cp: i32,
    pub pv: Vec<String>,
    pub collisions: u64,
    pub compactions: u32,
    // fração usada da arena de arestas (a que enche primeiro)
    pub hashfull: f32,
    // MultiPV > 1: uma linha por lance da raiz, do mais visitado para o menos
    pub lines: Vec<Line>,
}

pub struct Line {
    pub pv: Vec<String>,
    pub score_cp: i32,
    pub visits: u32,
}

// forma de um nó da árvore (ver Search::shape)
pub struct Step {
    // o lance mais visitado no nó
    pub mv: String,
    // visitas dos filhos do nó
    pub visits: u32,
    pub d: f32,
    pub h: f32,
    // prior e fração das visitas do lance mais visitado
    pub prior: f32,
    pub share: f32,
    // valor do lance na visão de quem joga no nó
    pub score_cp: i32,
    // lance de maior prior e o valor dele (None se a busca nunca o visitou)
    pub natural_mv: String,
    pub natural_cp: Option<i32>,
    // parte de d que vem do lance: visitas * log2(visitas / prior)
    pub surprise: f32,
}

// lance contraintuitivo achado na árvore (ver Search::findings)
pub struct Finding {
    // lances da raiz até o nó (vazio = o próprio lance da raiz)
    pub line: Vec<String>,
    pub step: Step,
    hash: u64,
}

pub struct SearchResult {
    pub best: Move,
    pub info: Info,
}

pub fn value_to_cp(v: f32) -> i32 {
    (CP_SCALE * v.clamp(-0.9999, 0.9999).atanh()).round() as i32
}

// buffers de uma thread, reaproveitados entre simulações
struct Scratch {
    board: Board,
    path: Vec<usize>,
    hashes: Vec<u64>,
    moves: Vec<Move>,
    priors: Vec<f32>,
}

pub struct Search<'a> {
    tree: &'a Tree,
    root: Board,
    // hashes das posições da partida até a raiz (inclusive), para a repetição
    history: Vec<u64>,
    eval: &'a dyn Evaluator,
    params: Params,
    seldepth: AtomicUsize,
    // maior PV já informada: o depth do info não volta (GUIs descartam linhas com depth menor que a anterior)
    depth: AtomicUsize,
    collisions: AtomicU64,
    // cada simulação segura a leitura; a compactação pega a escrita (para todas as threads)
    gate: std::sync::RwLock<()>,
    compactions: AtomicU32,
}

impl<'a> Search<'a> {
    pub fn new(
        tree: &'a Tree,
        root: Board,
        history: Vec<u64>,
        eval: &'a dyn Evaluator,
        params: Params,
    ) -> Self {
        tree.reset();
        Self::resume(tree, root, history, eval, params)
    }

    // continua a árvore que está no `tree` (sem limpar): a raiz tem de ser a mesma posição da busca anterior.
    // Usado na análise sem fim do navegador, que busca em fatias e responde entre uma e outra
    pub fn resume(
        tree: &'a Tree,
        root: Board,
        history: Vec<u64>,
        eval: &'a dyn Evaluator,
        params: Params,
    ) -> Self {
        Self {
            tree,
            root,
            history,
            eval,
            params,
            seldepth: AtomicUsize::new(0),
            depth: AtomicUsize::new(0),
            collisions: AtomicU64::new(0),
            gate: std::sync::RwLock::new(()),
            compactions: AtomicU32::new(0),
        }
    }

    fn root(&self) -> &Node {
        &self.tree.nodes[0]
    }

    pub fn run(
        &self,
        limits: &Limits,
        stop: &AtomicBool,
        mut on_info: impl FnMut(&Info),
    ) -> SearchResult {
        let start = Instant::now();
        let done = AtomicBool::new(false);
        let mut scratch = self.scratch();

        // a primeira simulação expande a raiz antes das outras threads começarem
        self.simulate(&mut scratch);
        // um lance legal só, jogando por tempo: não gasta o relógio, mas busca SINGLE_NODES nós para a avaliação (sem
        // busca ela saía 0, e as GUIs liam um xeque forçado como se a vantagem tivesse sumido)
        let single = self.root().num_edges.load(Relaxed) == 1 && limits.time.is_some();
        let capped = Limits {
            nodes: Some(limits.nodes.map_or(SINGLE_NODES, |n| n.min(SINGLE_NODES))),
            time: limits.time,
            depth: limits.depth,
        };
        let limits = if single { &capped } else { limits };
        if self.root().state.load(Acquire) == EXPANDED {
            std::thread::scope(|s| {
                for _ in 1..self.params.threads.max(1) {
                    s.spawn(|| {
                        let mut scratch = self.scratch();
                        while !self.should_stop(limits, stop, &done) {
                            self.step(&mut scratch);
                        }
                    });
                }
                // thread principal: tambem busca, controla o tempo e manda o info periodico
                let mut next_info = Duration::from_secs(1);
                let mut sims = 0u32;
                while !self.should_stop(limits, stop, &done) {
                    self.step(&mut scratch);
                    sims += 1;
                    if sims.is_multiple_of(16) {
                        let elapsed = start.elapsed();
                        if limits.time.is_some_and(|t| elapsed >= t) {
                            break;
                        }
                        if limits
                            .depth
                            .is_some_and(|d| sims.is_multiple_of(256) && self.depth_reached(d))
                        {
                            break;
                        }
                        if elapsed >= next_info {
                            on_info(&self.info(elapsed));
                            next_info += Duration::from_secs(1);
                        }
                    }
                }
                done.store(true, Relaxed);
            });
        }

        let info = self.info(start.elapsed());
        let best = self
            .best_edge(0)
            .map(|e| unpack(self.tree.edges[e].mv.load(Relaxed)))
            .unwrap_or_else(|| {
                // raiz terminal: devolve qualquer lance legal, se houver
                let mut any = None;
                self.root.generate_moves(|ms| {
                    any = ms.into_iter().next();
                    true
                });
                any.unwrap_or(Move {
                    from: Square::A1,
                    to: Square::A1,
                    promotion: None,
                })
            });
        SearchResult { best, info }
    }

    // limite de profundidade (go depth): a PV chegou a d lances, ou termina numa posição terminal (mate, empate) e
    // não vai mais crescer; o terminal só conta com visitas suficientes para a PV já ter assentado
    fn depth_reached(&self, d: usize) -> bool {
        let tree = self.tree;
        let (mut e, mut len) = (self.best_edge(0), 0);
        while let Some(edge) = e {
            let c = tree.edges[edge].child.load(Acquire) as usize;
            if c == 0 || tree.nodes[c].n.load(Relaxed) == 0 {
                break;
            }
            len += 1;
            if tree.nodes[c].state.load(Acquire) == TERMINAL
                && self.root().n.load(Relaxed) >= 20_000
            {
                return true;
            }
            e = self.best_edge(c);
        }
        self.depth.fetch_max(len, Relaxed).max(len) >= d
    }

    fn should_stop(&self, limits: &Limits, stop: &AtomicBool, done: &AtomicBool) -> bool {
        stop.load(Relaxed)
            || done.load(Relaxed)
            || limits
                .nodes
                .is_some_and(|l| self.root().n.load(Relaxed) as u64 >= l)
    }

    fn scratch(&self) -> Scratch {
        Scratch {
            board: self.root.clone(),
            path: Vec::with_capacity(256),
            hashes: self.history.clone(),
            moves: Vec::with_capacity(256),
            priors: Vec::with_capacity(256),
        }
    }

    // uma simulação; compacta a árvore antes se a arena estiver quase cheia
    fn step(&self, s: &mut Scratch) {
        let threads = self.params.threads.max(1);
        if self.tree.near_full(threads) {
            let _all = self.gate.write().unwrap();
            if self.tree.near_full(threads) {
                self.tree.compact();
                self.compactions.fetch_add(1, Relaxed);
            }
        }
        let _one = self.gate.read().unwrap();
        self.simulate(s);
    }

    // uma descida completa; se outra thread estiver expandindo a folha, desfaz o virtual loss e desiste (colisão)
    fn simulate(&self, s: &mut Scratch) {
        let tree = self.tree;
        s.board.clone_from(&self.root);
        s.hashes.truncate(self.history.len());
        s.path.clear();
        s.path.push(0);
        let mut idx = 0;
        let leaf = loop {
            let node = &tree.nodes[idx];
            match node.state.load(Acquire) {
                EXPANDED => {
                    let e = self.select(idx);
                    s.board
                        .play_unchecked(unpack(tree.edges[e].mv.load(Relaxed)));
                    s.hashes.push(s.board.hash());
                    let Some(c) = tree.child(e) else {
                        // arena de nós cheia: avalia a posição sem guardar nada
                        break Some(
                            self.terminal(false, s)
                                .unwrap_or_else(|| self.eval.value(&s.board)),
                        );
                    };
                    tree.nodes[c].vl.fetch_add(1, Relaxed);
                    s.path.push(c);
                    idx = c;
                }
                TERMINAL | FULL => break Some(node.fixed_value()),
                UNEXPANDED
                    if node
                        .state
                        .compare_exchange(UNEXPANDED, EXPANDING, Acquire, Relaxed)
                        .is_ok() =>
                {
                    break Some(self.first_visit(idx, s));
                }
                // segunda visita com expansão adiada: cria as arestas e continua descendo
                VISITED
                    if node
                        .state
                        .compare_exchange(VISITED, EXPANDING, Acquire, Relaxed)
                        .is_ok() =>
                {
                    self.legal_moves(s);
                    if !self.create_edges(idx, s) {
                        node.state.store(FULL, Release);
                        break Some(node.fixed_value());
                    }
                }
                _ => break None,
            }
        };
        self.seldepth
            .fetch_max(s.hashes.len() - self.history.len(), Relaxed);

        let Some(mut v) = leaf else {
            self.collisions.fetch_add(1, Relaxed);
            for &i in &s.path[1..] {
                tree.nodes[i].vl.fetch_sub(1, Relaxed);
            }
            std::hint::spin_loop();
            return;
        };
        // folha avaliada sem nó (arena cheia) fica um nivel abaixo do ultimo nó do caminho
        if s.hashes.len() - self.history.len() > s.path.len() - 1 {
            v = -v;
        }
        for (k, &i) in s.path.iter().enumerate().rev() {
            let node = &tree.nodes[i];
            node.w.fetch_add((v as f64 * W_SCALE) as i64, Relaxed);
            node.n.fetch_add(1, Relaxed);
            if k > 0 {
                node.vl.fetch_sub(1, Relaxed);
            }
            v = -v;
        }
    }

    // PUCT com virtual loss: descidas pendentes contam como derrota para quem escolhe o lance; devolve a aresta
    fn select(&self, idx: usize) -> usize {
        let params = &self.params;
        let parent = &self.tree.nodes[idx];
        let parent_n = parent.n.load(Relaxed);
        let sqrt_n = ((parent_n + parent.vl.load(Relaxed)) as f32).sqrt();
        let cpuct = params.cpuct
            + params.cpuct_factor
                * ((parent_n as f32 + params.cpuct_base) / params.cpuct_base).ln();
        let mut best = (f32::NEG_INFINITY, 0);
        for e in parent.edges() {
            let st = self.tree.stats(e);
            let visits = st.n + st.vl;
            // filho não visitado: q 0 (FPU absoluto)
            let q = if visits == 0 {
                0.0
            } else {
                (-(st.w + st.vl as f64) / visits as f64) as f32
            };
            let score = q + cpuct * self.tree.edges[e].prior() * sqrt_n / (1 + visits) as f32;
            if score > best.0 {
                best = (score, e);
            }
        }
        best.1
    }

    fn legal_moves(&self, s: &mut Scratch) {
        s.moves.clear();
        s.board.generate_moves(|ms| {
            s.moves.extend(ms);
            false
        });
    }

    // valor exato se a posição em s.board for terminal; deixa os lances legais em s.moves
    fn terminal(&self, is_root: bool, s: &mut Scratch) -> Option<f32> {
        let board = &s.board;
        // empates (nunca na raiz): repetição dentro da janela dos 50 lances, regra dos 50 lances, material insuficiente
        if !is_root {
            let window = board.halfmove_clock() as usize;
            let (current, previous) = s.hashes.split_last().unwrap();
            let repeated = previous
                .iter()
                .rev()
                .take(window)
                .skip(1)
                .step_by(2)
                .any(|h| h == current);
            if repeated || board.halfmove_clock() >= 100 || insufficient_material(board) {
                return Some(0.0);
            }
        }
        self.legal_moves(s);
        s.moves.is_empty().then(|| {
            if s.board.checkers().is_empty() {
                0.0
            } else {
                -1.0
            }
        })
    }

    // chamada por quem ganhou o EXPANDING de um nó novo: terminal e value (arestas so na raiz;
    // os outros nós as ganham na segunda visita, expansão adiada);
    // publica o estado e devolve o valor na visão de quem joga no nó
    fn first_visit(&self, idx: usize, s: &mut Scratch) -> f32 {
        let node = &self.tree.nodes[idx];
        if let Some(v) = self.terminal(idx == 0, s) {
            node.v.store(v.to_bits(), Relaxed);
            node.state.store(TERMINAL, Release);
            return v;
        }
        let v = self.eval.value(&s.board);
        node.v.store(v.to_bits(), Relaxed);
        if idx != 0 {
            node.state.store(VISITED, Release);
        } else if !self.create_edges(idx, s) {
            node.state.store(FULL, Release);
        }
        v
    }

    // calcula os priors dos lances em s.moves, cria as arestas e publica EXPANDED; false se a arena encheu
    fn create_edges(&self, idx: usize, s: &mut Scratch) -> bool {
        let node = &self.tree.nodes[idx];
        let Some(first) = bump(&self.tree.edge_cursor, s.moves.len(), self.tree.edges.len()) else {
            return false;
        };
        s.priors.clear();
        s.priors.resize(s.moves.len(), 0.0);
        self.eval.priors(&s.board, &s.moves, &mut s.priors);
        for (k, (&m, &p)) in s.moves.iter().zip(&s.priors).enumerate() {
            let edge = &self.tree.edges[first + k];
            edge.child.store(0, Relaxed);
            edge.mv.store(pack(m), Relaxed);
            edge.p.store((p * P_SCALE).round() as u16, Relaxed);
        }
        node.first_edge.store(first as u32, Relaxed);
        node.num_edges.store(s.moves.len() as u8, Relaxed);
        node.state.store(EXPANDED, Release);
        true
    }

    // aresta mais visitada; empate decidido pelo valor
    fn best_edge(&self, idx: usize) -> Option<usize> {
        let node = &self.tree.nodes[idx];
        if node.state.load(Acquire) != EXPANDED {
            return None;
        }
        node.edges().max_by(|&a, &b| {
            let (sa, sb) = (self.tree.stats(a), self.tree.stats(b));
            sa.n.cmp(&sb.n)
                .then(sa.q_parent().total_cmp(&sb.q_parent()))
        })
    }

    // perfil instrumental: retrato da árvore ao longo da variante principal
    // para medir a forma da busca (quanto ela concordou com a policy, quão concentradas ficaram as visitas)
    // Um bloco por nó da PV (seguindo o filho mais visitado), separados por ';', no formato "N{lance:n:P:Q:V|...}":
    // N = visitas do nó (na raiz, sem a visita que a expandiu); para cada filho, com o mais visitado primeiro:
    // visitas n, prior P, valor médio Q e avaliação da value V, os dois na visão de quem escolhe o lance,
    // '-' quando ausentes. Devolve "-" se não há PV
    pub fn instrumental_profile(&self) -> String {
        let tree = self.tree;
        let mut board = self.root.clone();
        let mut idx = 0;
        let mut out = Vec::new();
        for ply in 0..256 {
            let node = &tree.nodes[idx];
            if node.state.load(Acquire) != EXPANDED {
                break;
            }
            let edges: Vec<usize> = node.edges().collect();
            // empate de visitas: fica o primeiro na ordem das arestas
            let Some(selected) = (0..edges.len())
                .filter(|&k| tree.stats(edges[k]).n > 0)
                .fold(None, |best: Option<usize>, k| match best {
                    Some(b) if tree.stats(edges[b]).n >= tree.stats(edges[k]).n => Some(b),
                    _ => Some(k),
                })
            else {
                break;
            };
            let mut order: Vec<usize> = (0..edges.len()).collect();
            order[..=selected].rotate_right(1);

            let children: Vec<String> = order
                .iter()
                .map(|&k| {
                    let e = edges[k];
                    let edge = &tree.edges[e];
                    let mv = display_uci_move(&board, unpack(edge.mv.load(Relaxed)));
                    let st = tree.stats(e);
                    let q = if st.n > 0 {
                        format!("{}", -st.w / st.n as f64)
                    } else {
                        "-".into()
                    };
                    let c = edge.child.load(Acquire) as usize;
                    // V: avaliação da rede no filho (não existe para terminais)
                    let v = match (st.n, c) {
                        (n, c)
                            if n > 0
                                && c != 0
                                && matches!(
                                    tree.nodes[c].state.load(Acquire),
                                    EXPANDED | VISITED | FULL
                                ) =>
                        {
                            format!("{}", -tree.nodes[c].fixed_value())
                        }
                        _ => "-".into(),
                    };
                    format!("{mv}:{}:{}:{q}:{v}", st.n, edge.prior())
                })
                .collect();
            let n = node.n.load(Relaxed);
            let n = if ply == 0 { n.saturating_sub(1) } else { n };
            out.push(format!("{n}{{{}}}", children.join("|")));

            let e = edges[selected];
            board.play_unchecked(unpack(tree.edges[e].mv.load(Relaxed)));
            idx = tree.edges[e].child.load(Acquire) as usize;
        }
        if out.is_empty() {
            "-".into()
        } else {
            out.join(";")
        }
    }

    // lances contraintuitivos na árvore: nós em que a busca concentrou as visitas num lance que a policy quase não
    // considerava (prior baixo, maioria das visitas). Percorre as linhas principais (filhos com pelo menos 10% das
    // visitas do pai) enquanto os nós têm min_visits ou mais. Para cada nó calcula também, sobre os filhos visitados,
    // H = entropia das visitas e d = divergência KL das visitas em relação aos priors (em bits): d alto = a busca
    // discordou da policy; H baixo = convergiu num lance.
    // Devolve do mais surpreendente para o menos (surpresa = visitas * log2(visitas / prior), a parte de d que vem do
    // lance), uma vez por posição (a mesma posição alcançada por ordens de lances diferentes fica com a mais visitada)
    pub fn findings(&self, min_visits: u32) -> Vec<Finding> {
        let mut all = Vec::new();
        let mut line = Vec::new();
        self.findings_from(0, self.root.clone(), &mut line, min_visits, &mut all);
        all.sort_by(|a, b| b.step.visits.cmp(&a.step.visits));
        let mut seen = std::collections::HashSet::new();
        let mut out: Vec<Finding> = all.into_iter().filter(|f| seen.insert(f.hash)).collect();
        out.sort_by(|a, b| b.step.surprise.total_cmp(&a.step.surprise));
        out
    }

    // forma de um nó expandido, sobre os filhos visitados: visitas, H = entropia das visitas e d = divergência KL das
    // visitas em relação aos priors (em bits), e o lance mais visitado; devolve também a aresta desse lance
    fn shape(&self, idx: usize, board: &Board) -> Option<(Step, usize)> {
        let tree = self.tree;
        let node = &tree.nodes[idx];
        if node.state.load(Acquire) != EXPANDED {
            return None;
        }
        let edges: Vec<usize> = node.edges().collect();
        let stats: Vec<Stats> = edges.iter().map(|&e| tree.stats(e)).collect();
        let total: u32 = stats.iter().map(|s| s.n).sum();
        if total == 0 {
            return None;
        }
        let psum: f32 = edges.iter().map(|&e| tree.edges[e].prior().max(1e-6)).sum();
        let (mut h, mut d) = (0.0f32, 0.0f32);
        for (k, st) in stats.iter().enumerate().filter(|(_, st)| st.n > 0) {
            let pi = st.n as f32 / total as f32;
            h -= pi * pi.log2();
            d += pi * (pi / (tree.edges[edges[k]].prior().max(1e-6) / psum)).log2();
        }
        let best = (0..edges.len()).max_by_key(|&k| stats[k].n).unwrap();
        let (prior, share) = (
            tree.edges[edges[best]].prior(),
            stats[best].n as f32 / total as f32,
        );
        // lance natural: o de maior prior (o que a policy jogaria)
        let natural = (0..edges.len())
            .max_by(|&a, &b| {
                tree.edges[edges[a]]
                    .prior()
                    .total_cmp(&tree.edges[edges[b]].prior())
            })
            .unwrap();
        let step = Step {
            natural_mv: display_uci_move(
                board,
                unpack(tree.edges[edges[natural]].mv.load(Relaxed)),
            )
            .to_string(),
            natural_cp: (stats[natural].n > 0).then(|| value_to_cp(stats[natural].q_parent())),
            mv: display_uci_move(board, unpack(tree.edges[edges[best]].mv.load(Relaxed)))
                .to_string(),
            visits: total,
            d: d.max(0.0),
            h: h.max(0.0),
            prior,
            share,
            score_cp: value_to_cp(stats[best].q_parent()),
            surprise: share * (share / (prior.max(1e-6) / psum)).log2(),
        };
        Some((step, edges[best]))
    }

    // descritor sequencial: a forma de cada nó da variante principal (seguindo o lance mais visitado), da raiz até o
    // primeiro nó com menos de min_visits visitas (mais fundo os números são ruído: com poucas visitas quase sempre
    // um lance fica com a maioria, H sai baixo e d alto)
    pub fn sequence(&self, min_visits: u32) -> Vec<Step> {
        let mut out = Vec::new();
        let (mut idx, mut board) = (0, self.root.clone());
        while let Some((step, e)) = self.shape(idx, &board) {
            if step.visits < min_visits {
                break;
            }
            out.push(step);
            board.play_unchecked(unpack(self.tree.edges[e].mv.load(Relaxed)));
            idx = self.tree.edges[e].child.load(Acquire) as usize;
            if idx == 0 {
                break;
            }
        }
        out
    }

    fn findings_from(
        &self,
        idx: usize,
        board: Board,
        line: &mut Vec<String>,
        min_visits: u32,
        out: &mut Vec<Finding>,
    ) {
        const PRIOR_MAX: f32 = 0.10;
        const SHARE_MIN: f32 = 0.5;
        const BRANCH_MIN: f32 = 0.10;
        // mais fundo a policy erra demais para o sinal dizer algo
        const MAX_DEPTH: usize = 10;
        let tree = self.tree;
        if line.len() >= MAX_DEPTH {
            return;
        }
        let Some((step, _)) = self.shape(idx, &board) else {
            return;
        };
        if step.visits < min_visits {
            return;
        }
        let total = step.visits;
        if step.prior <= PRIOR_MAX && step.share >= SHARE_MIN {
            out.push(Finding {
                line: line.clone(),
                step,
                hash: board.hash(),
            });
        }
        for e in tree.nodes[idx].edges() {
            let child = tree.edges[e].child.load(Acquire) as usize;
            let n = tree.stats(e).n;
            if child == 0 || n < min_visits || (n as f32) < BRANCH_MIN * total as f32 {
                continue;
            }
            let mv = unpack(tree.edges[e].mv.load(Relaxed));
            let mut next = board.clone();
            next.play_unchecked(mv);
            line.push(display_uci_move(&board, mv).to_string());
            self.findings_from(child, next, line, min_visits, out);
            line.pop();
        }
    }

    // PV a partir da aresta e (o lance dela e depois o filho mais visitado a cada nó)
    fn pv_from(&self, first: usize) -> Vec<String> {
        let tree = self.tree;
        let mut pv = Vec::new();
        let mut board = self.root.clone();
        let mut e = Some(first);
        while let Some(edge) = e {
            let c = tree.edges[edge].child.load(Acquire) as usize;
            if c == 0 || tree.nodes[c].n.load(Relaxed) == 0 {
                break;
            }
            let mv = unpack(tree.edges[edge].mv.load(Relaxed));
            pv.push(display_uci_move(&board, mv).to_string());
            board.play_unchecked(mv);
            e = self.best_edge(c);
        }
        pv
    }

    fn info(&self, elapsed: Duration) -> Info {
        let tree = self.tree;
        let pv = self
            .best_edge(0)
            .map(|e| self.pv_from(e))
            .unwrap_or_default();
        let score = self.best_edge(0).map_or(0.0, |e| tree.stats(e).q_parent());
        let mut lines = Vec::new();
        if self.params.multipv > 1 && self.root().state.load(Acquire) == EXPANDED {
            // sempre min(MultiPV, lances legais) linhas: o En Croissant espera esse número exato. Os lances sem
            // visita vêm depois, pelo prior, com o lance como PV e a nota da pior linha visitada
            let mut order: Vec<usize> = self.root().edges().collect();
            order.sort_by(|&a, &b| {
                let (sa, sb) = (tree.stats(a), tree.stats(b));
                sb.n.cmp(&sa.n)
                    .then(sb.q_parent().total_cmp(&sa.q_parent()))
                    .then(tree.edges[b].prior().total_cmp(&tree.edges[a].prior()))
            });
            for &e in order.iter().take(self.params.multipv) {
                let st = tree.stats(e);
                let line = if st.n > 0 {
                    Line {
                        pv: self.pv_from(e),
                        score_cp: value_to_cp(st.q_parent()),
                        visits: st.n,
                    }
                } else {
                    let mv = unpack(tree.edges[e].mv.load(Relaxed));
                    let worst = lines
                        .last()
                        .map_or(value_to_cp(score), |l: &Line| l.score_cp);
                    Line {
                        pv: vec![display_uci_move(&self.root, mv).to_string()],
                        score_cp: worst,
                        visits: 0,
                    }
                };
                lines.push(line);
            }
        }
        Info {
            depth: self.depth.fetch_max(pv.len(), Relaxed).max(pv.len()),
            seldepth: self.seldepth.load(Relaxed),
            nodes: self.root().n.load(Relaxed) as u64,
            elapsed,
            score_cp: value_to_cp(score),
            pv,
            collisions: self.collisions.load(Relaxed),
            compactions: self.compactions.load(Relaxed),
            hashfull: (tree.edge_cursor.load(Relaxed).min(tree.edges.len()) as f32
                / tree.edges.len() as f32)
                .max(
                    tree.node_cursor.load(Relaxed).min(tree.nodes.len()) as f32
                        / tree.nodes.len() as f32,
                ),
            lines,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use olidra_core::eval::{MaterialEvaluator, UniformEvaluator};

    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    fn run(
        tree: &Tree,
        board: &Board,
        history: Vec<u64>,
        eval: &dyn Evaluator,
        threads: usize,
        nodes: u64,
    ) -> SearchResult {
        let s = Search::new(
            tree,
            board.clone(),
            history,
            eval,
            Params {
                threads,
                ..Params::default()
            },
        );
        s.run(
            &Limits {
                nodes: Some(nodes),
                time: None,
                depth: None,
            },
            &AtomicBool::new(false),
            |_| {},
        )
    }

    fn best(fen: &str, eval: &dyn Evaluator, threads: usize, nodes: u64) -> String {
        let board = Board::from_fen(fen, false).unwrap();
        let r = run(
            &Tree::new(64 << 20),
            &board,
            vec![board.hash()],
            eval,
            threads,
            nodes,
        );
        display_uci_move(&board, r.best).to_string()
    }

    fn used_nodes(tree: &Tree) -> &[Node] {
        &tree.nodes[..tree.node_cursor.load(Relaxed).min(tree.nodes.len())]
    }

    #[test]
    fn tamanhos() {
        assert_eq!(size_of::<Node>(), 32);
        assert_eq!(size_of::<Edge>(), 8);
    }

    #[test]
    fn acha_mate_em_1() {
        // mate do corredor, nas duas cores, com 1 e 4 threads
        for threads in [1, 4] {
            assert_eq!(
                best(
                    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
                    &UniformEvaluator,
                    threads,
                    2000
                ),
                "a1a8"
            );
            assert_eq!(
                best(
                    "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1",
                    &UniformEvaluator,
                    threads,
                    2000
                ),
                "a8a1"
            );
        }
    }

    #[test]
    fn acha_mate_de_dama_com_material() {
        // Db8# (a torre em b1 protege a dama)
        assert_eq!(
            best(
                "6k1/5ppp/8/8/8/8/1Q3PPP/1R4K1 w - - 0 1",
                &MaterialEvaluator,
                1,
                5000
            ),
            "b2b8"
        );
    }

    #[test]
    fn repeticao_conta_como_empate() {
        // raiz depois de 1.Cf3 Cf6: na arvore, 2.Cg1 Cg8 repete a posição inicial da partida
        let mut board = Board::default();
        let mut history = vec![board.hash()];
        for mv in ["g1f3", "g8f6"] {
            board.play(mv.parse().unwrap());
            history.push(board.hash());
        }
        let tree = Tree::new(64 << 20);
        run(&tree, &board, history, &MaterialEvaluator, 1, 20000);
        // sem captura possivel de material decisivo, o unico terminal com valor 0 é a repetição
        assert!(
            used_nodes(&tree)
                .iter()
                .any(|n| n.state.load(Relaxed) == TERMINAL && n.fixed_value() == 0.0)
        );
    }

    // invariantes no fim da busca, com 1 e com varias threads
    #[test]
    fn visitas_somam_e_virtual_loss_zera() {
        let board = Board::from_fen(KIWIPETE, false).unwrap();
        for threads in [1, 8] {
            let tree = Tree::new(64 << 20);
            run(
                &tree,
                &board,
                vec![board.hash()],
                &MaterialEvaluator,
                threads,
                20000,
            );
            for node in used_nodes(&tree) {
                assert_eq!(node.vl.load(Relaxed), 0);
                if node.state.load(Relaxed) == EXPANDED {
                    let filhos: u32 = node.edges().map(|e| tree.stats(e).n).sum();
                    assert_eq!(filhos, node.n.load(Relaxed) - 1, "{threads} threads");
                }
            }
            assert!(tree.nodes[0].n.load(Relaxed) >= 20000);
        }
    }

    #[test]
    fn uma_thread_e_deterministica() {
        let board = Board::from_fen(KIWIPETE, false).unwrap();
        let visitas = || {
            let tree = Tree::new(64 << 20);
            run(
                &tree,
                &board,
                vec![board.hash()],
                &MaterialEvaluator,
                1,
                5000,
            );
            tree.nodes[0]
                .edges()
                .map(|e| tree.stats(e).n)
                .collect::<Vec<_>>()
        };
        assert_eq!(visitas(), visitas());
    }

    // arena minuscula: as duas arenas enchem e a busca continua sem travar, com as visitas ainda somando
    #[test]
    fn arvore_cheia_nao_trava() {
        let board = Board::default();
        for threads in [1, 4] {
            let tree = Tree::new(64 * (size_of::<Node>() + EDGES_PER_NODE * size_of::<Edge>()));
            let r = run(
                &tree,
                &board,
                vec![board.hash()],
                &MaterialEvaluator,
                threads,
                20000,
            );
            assert!(board.is_legal(r.best));
            assert!(tree.nodes[0].n.load(Relaxed) >= 20000);
            for node in used_nodes(&tree) {
                assert_eq!(node.vl.load(Relaxed), 0);
            }
        }
    }

    // regras do formato do perfil: filhos visitados têm Q, não visitados não têm Q nem V,
    // o primeiro filho é o mais visitado e a soma das visitas dos filhos bate com N do nó
    #[test]
    fn perfil_instrumental_valido() {
        let board = Board::from_fen(KIWIPETE, false).unwrap();
        let tree = Tree::new(64 << 20);
        let s = Search::new(
            &tree,
            board.clone(),
            vec![board.hash()],
            &MaterialEvaluator,
            Params::default(),
        );
        s.run(
            &Limits {
                nodes: Some(5000),
                time: None,
                depth: None,
            },
            &AtomicBool::new(false),
            |_| {},
        );
        let profile = s.instrumental_profile();
        let nodes: Vec<&str> = profile.split(';').collect();
        assert!(nodes.len() >= 2, "{profile}");
        for (ply, node) in nodes.iter().enumerate() {
            let (n, rest) = node.split_once('{').unwrap();
            let n: u32 = n.parse().unwrap();
            let children: Vec<Vec<&str>> = rest
                .trim_end_matches('}')
                .split('|')
                .map(|c| c.split(':').collect())
                .collect();
            let visits: Vec<u32> = children.iter().map(|c| c[1].parse().unwrap()).collect();
            assert_eq!(visits[0], *visits.iter().max().unwrap());
            // raiz ja sem a visita da expansão; nos outros nós, a primeira visita não desce para um filho
            assert_eq!(visits.iter().sum::<u32>(), if ply == 0 { n } else { n - 1 });
            for c in &children {
                assert_eq!(c.len(), 5);
                let visited = c[1] != "0";
                assert_eq!(c[3] != "-", visited);
                assert!(visited || c[4] == "-");
                if visited {
                    assert!(c[3].parse::<f64>().unwrap().abs() <= 1.000001);
                }
            }
        }
        // a raiz lista todos os lances legais, em UCI padrão (roque e1g1)
        let mut legal = 0;
        board.generate_moves(|ms| {
            legal += ms.len();
            false
        });
        assert_eq!(nodes[0].matches('|').count() + 1, legal);
        assert!(nodes[0].contains("e1g1:") && !nodes[0].contains("e1h1:"));
    }

    // a mesma arena serve para buscas seguidas
    #[test]
    fn arena_reaproveitada() {
        let tree = Tree::new(16 << 20);
        let board = Board::from_fen(KIWIPETE, false).unwrap();
        for _ in 0..3 {
            let r = run(
                &tree,
                &board,
                vec![board.hash()],
                &MaterialEvaluator,
                2,
                3000,
            );
            assert!(board.is_legal(r.best));
        }
    }
}
