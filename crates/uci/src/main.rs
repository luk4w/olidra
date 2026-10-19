use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use olidra_core::cozy_chess::util::{display_uci_move, parse_uci_move};
use olidra_core::cozy_chess::{Board, Color};
use olidra_core::eval::Evaluator;
use olidra_core::{NAME, VERSION};
use olidra_nnue::{NnueEvaluator, PolicyNet, ValueNet};
use olidra_search::{Info, Limits, Params, Search, Tree};

struct Options {
    hash_mb: usize,
    threads: usize,
    profile: bool,
    // lances contraintuitivos achados na árvore, no fim da busca
    findings: bool,
    cpuct: f32,
    cpuct_base: f32,
    cpuct_factor: f32,
    eval_file: String,
    policy_file: String,
    move_overhead: u64,
    multipv: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            hash_mb: 1024,
            threads: 1,
            profile: false,
            findings: false,
            cpuct: 1.75,
            cpuct_base: 38739.0,
            cpuct_factor: 3.89,
            eval_file: "nets/value.bin".into(),
            policy_file: "nets/policy.bin".into(),
            move_overhead: 30,
            multipv: 1,
        }
    }
}

// caminho relativo: primeiro à pasta do executavel, depois à raiz do projeto (target/release/olidra.exe -> ../..),
// por ultimo ao diretorio atual (as GUIs abrem o motor com um diretorio atual qualquer)
fn resolve(path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        return p.into();
    }
    let dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    let candidates = dir
        .iter()
        .flat_map(|d| [d.join(p), d.join("../..").join(p)]);
    candidates
        .into_iter()
        .find(|q| q.exists())
        .unwrap_or_else(|| p.into())
}

// value e policy em int8 (quantizadas ao carregar); sem value usa material, sem policy usa prior uniforme
fn build_evaluator(o: &Options, out: &mut impl Write) -> Arc<dyn Evaluator> {
    let value = match ValueNet::load(&resolve(&o.eval_file)) {
        Ok(v) => Some(v),
        Err(e) => {
            writeln!(
                out,
                "info string erro ao ler a value ({e}); usando material"
            )
            .unwrap();
            None
        }
    };
    let policy = match PolicyNet::load(&resolve(&o.policy_file)) {
        Ok(p) => Some(p),
        Err(e) => {
            writeln!(
                out,
                "info string erro ao ler a policy ({e}); usando prior uniforme"
            )
            .unwrap();
            None
        }
    };
    Arc::new(NnueEvaluator { value, policy })
}

fn print_info(info: &Info) {
    let ms = info.elapsed.as_millis().max(1);
    let nps = info.nodes as u128 * 1000 / ms;
    let mut out = io::stdout().lock();
    if !info.lines.is_empty() {
        // MultiPV: uma linha por lance da raiz, do mais visitado para o menos
        for (k, l) in info.lines.iter().enumerate() {
            writeln!(
                out,
                "info depth {} seldepth {} multipv {} score cp {} nodes {} nps {nps} hashfull {} time {ms} pv {}",
                info.depth,
                info.seldepth,
                k + 1,
                l.score_cp,
                info.nodes,
                (info.hashfull * 1000.0) as u32,
                l.pv.join(" ")
            )
            .unwrap();
        }
        let visits: Vec<String> = info
            .lines
            .iter()
            .map(|l| format!("{}:{}", l.pv.first().map_or("-", |m| m.as_str()), l.visits))
            .collect();
        writeln!(out, "info string visitas {}", visits.join(" ")).unwrap();
        out.flush().unwrap();
        return;
    }
    writeln!(
        out,
        "info depth {} seldepth {} nodes {} nps {nps} hashfull {} time {ms} score cp {} pv {}",
        info.depth,
        info.seldepth,
        info.nodes,
        (info.hashfull * 1000.0) as u32,
        info.score_cp,
        info.pv.join(" ")
    )
    .unwrap();
    out.flush().unwrap();
}

// tempo do lance: fração do relogio + incremento, com teto de 40% do restante
fn time_for_move(tokens: &[&str], stm: Color, overhead: u64) -> Option<Duration> {
    let get = |key: &str| {
        tokens
            .iter()
            .position(|&t| t == key)
            .and_then(|i| tokens.get(i + 1))
            .and_then(|v| v.parse::<i64>().ok())
    };
    if let Some(mt) = get("movetime") {
        return Some(Duration::from_millis((mt - overhead as i64).max(1) as u64));
    }
    let (time, inc) = match stm {
        Color::White => (get("wtime")?, get("winc").unwrap_or(0)),
        Color::Black => (get("btime")?, get("binc").unwrap_or(0)),
    };
    let mtg = get("movestogo").unwrap_or(30).max(1);
    let target = time / mtg + inc * 3 / 4 - overhead as i64;
    let target = target.min(time * 2 / 5).max(1);
    Some(Duration::from_millis(target as u64))
}

struct Engine {
    options: Options,
    evaluator: Option<Arc<dyn Evaluator>>,
    // arena da arvore, criada uma vez e refeita so quando o Hash muda
    tree: Option<(usize, Arc<Tree>)>,
    board: Board,
    history: Vec<u64>,
    stop: Arc<AtomicBool>,
    searching: Option<JoinHandle<()>>,
}

impl Engine {
    fn wait(&mut self) {
        if let Some(h) = self.searching.take() {
            h.join().unwrap();
        }
    }

    fn evaluator(&mut self, out: &mut impl Write) -> Arc<dyn Evaluator> {
        if self.evaluator.is_none() {
            self.evaluator = Some(build_evaluator(&self.options, out));
        }
        self.evaluator.clone().unwrap()
    }

    fn tree(&mut self) -> Arc<Tree> {
        let key = self.options.hash_mb;
        if self.tree.as_ref().is_none_or(|(k, _)| *k != key) {
            self.tree = None; // libera a arena antiga antes de alocar a nova
            self.tree = Some((key, Arc::new(Tree::new(key * 1024 * 1024))));
        }
        self.tree.as_ref().unwrap().1.clone()
    }

    fn position(&mut self, tokens: &[&str]) {
        let moves_at = tokens
            .iter()
            .position(|&t| t == "moves")
            .unwrap_or(tokens.len());
        let board = match tokens.get(1) {
            Some(&"startpos") => Some(Board::default()),
            Some(&"fen") => Board::from_fen(&tokens[2..moves_at].join(" "), false).ok(),
            _ => None,
        };
        let Some(mut board) = board else { return };
        let mut history = vec![board.hash()];
        for m in tokens.iter().skip(moves_at + 1) {
            match parse_uci_move(&board, m) {
                Ok(mv) if board.is_legal(mv) => board.play_unchecked(mv),
                _ => {
                    println!("info string lance ilegal: {m}");
                    break;
                }
            }
            history.push(board.hash());
        }
        self.board = board;
        self.history = history;
    }

    fn go(&mut self, tokens: &[&str], out: &mut impl Write) {
        self.wait();
        let eval = self.evaluator(out);
        let nodes = tokens
            .iter()
            .position(|&t| t == "nodes")
            .and_then(|i| tokens.get(i + 1))
            .and_then(|v| v.parse().ok());
        let time = time_for_move(
            tokens,
            self.board.side_to_move(),
            self.options.move_overhead,
        );
        let depth = tokens
            .iter()
            .position(|&t| t == "depth")
            .and_then(|i| tokens.get(i + 1))
            .and_then(|v| v.parse().ok());
        let limits = Limits { nodes, time, depth };
        let o = &self.options;
        let params = Params {
            cpuct: o.cpuct,
            cpuct_base: o.cpuct_base,
            cpuct_factor: o.cpuct_factor,
            threads: o.threads,
            multipv: o.multipv,
        };
        let tree = self.tree();
        let (board, history, profile, findings) = (
            self.board.clone(),
            self.history.clone(),
            self.options.profile,
            self.options.findings,
        );
        self.stop.store(false, Ordering::Relaxed);
        let stop = self.stop.clone();
        self.searching = Some(std::thread::spawn(move || {
            let search = Search::new(&tree, board.clone(), history, eval.as_ref(), params);
            let result = search.run(&limits, &stop, print_info);
            print_info(&result.info);
            if profile {
                println!(
                    "info string instrumental_profile {}",
                    search.instrumental_profile()
                );
            }
            if findings {
                // so nós com visitas suficientes para o sinal não ser ruído
                let min_visits = (result.info.nodes / 256).max(256) as u32;
                // os 5 mais surpreendentes
                for f in search.findings(min_visits).into_iter().take(5) {
                    let st = &f.step;
                    println!(
                        "info string achado ply {} linha {} lance {} prior {:.3} visitas {:.3} d {:.2} H {:.2} cp {} n {}",
                        f.line.len(),
                        if f.line.is_empty() {
                            "-".into()
                        } else {
                            f.line.join(",")
                        },
                        st.mv,
                        st.prior,
                        st.share,
                        st.d,
                        st.h,
                        st.score_cp,
                        st.visits
                    );
                }
                // descritor sequencial ao longo da PV, um nó por bloco separado por ';':
                // lance:visitas:d:H:prior:fração das visitas:cp do lance:lance natural (maior prior):cp dele ('-' sem visitas)
                let seq: Vec<String> = search
                    .sequence(min_visits)
                    .iter()
                    .map(|st| {
                        let nat = st.natural_cp.map_or("-".into(), |c| c.to_string());
                        format!(
                            "{}:{}:{:.2}:{:.2}:{:.3}:{:.3}:{}:{}:{nat}",
                            st.mv,
                            st.visits,
                            st.d,
                            st.h,
                            st.prior,
                            st.share,
                            st.score_cp,
                            st.natural_mv
                        )
                    })
                    .collect();
                println!(
                    "info string sequencia {}",
                    if seq.is_empty() {
                        "-".into()
                    } else {
                        seq.join(";")
                    }
                );
            }
            let (nodes, edges) = tree.usage();
            println!(
                "info string colisoes {} nos {nodes} arestas {edges} compactacoes {}",
                result.info.collisions, result.info.compactions
            );
            println!("bestmove {}", display_uci_move(&board, result.best));
        }));
    }

    fn setoption(&mut self, line: &str) {
        let Some(rest) = line.split_once(" name ").map(|(_, r)| r) else {
            return;
        };
        let (name, value) = rest.split_once(" value ").unwrap_or((rest, ""));
        let (name, value) = (name.trim(), value.trim());
        let o = &mut self.options;
        match name.to_ascii_lowercase().as_str() {
            "hash" => o.hash_mb = value.parse().unwrap_or(o.hash_mb),
            "threads" => {
                o.threads = value.parse().unwrap_or(o.threads);
                return;
            }
            "instrumentalprofile" => {
                o.profile = value.eq_ignore_ascii_case("true");
                return;
            }
            "findings" => {
                o.findings = value.eq_ignore_ascii_case("true");
                return;
            }
            "cpuct" | "cpuctbase" | "cpuctfactor" => {
                let v: Option<f32> = value.parse().ok();
                match name.to_ascii_lowercase().as_str() {
                    "cpuct" => o.cpuct = v.map_or(o.cpuct, |v| v / 100.0),
                    "cpuctbase" => o.cpuct_base = v.unwrap_or(o.cpuct_base).max(1.0),
                    _ => o.cpuct_factor = v.map_or(o.cpuct_factor, |v| v / 100.0),
                }
                return;
            }
            "moveoverhead" => o.move_overhead = value.parse().unwrap_or(o.move_overhead),
            "evalfile" => o.eval_file = value.into(),
            "policyfile" => o.policy_file = value.into(),
            "multipv" => {
                o.multipv = value.parse::<usize>().unwrap_or(o.multipv).clamp(1, 50);
                return;
            }
            _ => return,
        }
        // as redes são recarregadas no proximo isready/go
        self.evaluator = None;
    }
}

fn main() {
    let stdin = io::stdin();
    let mut engine = Engine {
        options: Options::default(),
        evaluator: None,
        tree: None,
        board: Board::default(),
        history: vec![Board::default().hash()],
        stop: Arc::new(AtomicBool::new(false)),
        searching: None,
    };

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        // o pipe do PowerShell 5.1 manda um BOM antes da primeira linha
        let line = line.trim_start_matches('\u{feff}').trim();
        let tokens: Vec<&str> = line.split_whitespace().collect();
        // o lock do stdout fica so dentro de cada comando: a thread de busca imprime o bestmove
        match tokens.first().copied() {
            Some("uci") => {
                let mut out = io::stdout().lock();
                let d = Options::default();
                writeln!(out, "id name {NAME} {VERSION}").unwrap();
                writeln!(out, "id author luk4w").unwrap();
                writeln!(
                    out,
                    "option name Hash type spin default {} min 16 max 65536",
                    d.hash_mb
                )
                .unwrap();
                writeln!(
                    out,
                    "option name Threads type spin default {} min 1 max 256",
                    d.threads
                )
                .unwrap();
                writeln!(
                    out,
                    "option name InstrumentalProfile type check default {}",
                    d.profile
                )
                .unwrap();
                writeln!(
                    out,
                    "option name Findings type check default {}",
                    d.findings
                )
                .unwrap();
                writeln!(
                    out,
                    "option name CPuct type spin default {} min 1 max 1000",
                    (d.cpuct * 100.0) as i32
                )
                .unwrap();
                writeln!(
                    out,
                    "option name CPuctBase type spin default {} min 1 max 10000000",
                    d.cpuct_base as i32
                )
                .unwrap();
                writeln!(
                    out,
                    "option name CPuctFactor type spin default {} min 0 max 1000",
                    (d.cpuct_factor * 100.0) as i32
                )
                .unwrap();
                writeln!(
                    out,
                    "option name MoveOverhead type spin default {} min 0 max 5000",
                    d.move_overhead
                )
                .unwrap();
                writeln!(
                    out,
                    "option name EvalFile type string default {}",
                    d.eval_file
                )
                .unwrap();
                writeln!(
                    out,
                    "option name PolicyFile type string default {}",
                    d.policy_file
                )
                .unwrap();
                writeln!(
                    out,
                    "option name MultiPV type spin default {} min 1 max 50",
                    d.multipv
                )
                .unwrap();
                writeln!(out, "uciok").unwrap();
            }
            Some("isready") => {
                let mut out = io::stdout().lock();
                engine.evaluator(&mut out);
                engine.tree();
                writeln!(out, "readyok").unwrap();
            }
            Some("setoption") => engine.setoption(line),
            Some("ucinewgame") => engine.wait(),
            Some("position") => {
                engine.wait();
                engine.position(&tokens);
            }
            Some("go") => engine.go(&tokens, &mut io::stdout()),
            Some("stop") => {
                engine.stop.store(true, Ordering::Relaxed);
                engine.wait();
            }
            Some("quit") => {
                engine.stop.store(true, Ordering::Relaxed);
                engine.wait();
                break;
            }
            _ => {}
        }
        io::stdout().flush().unwrap();
    }
}
