# Olidra

<table>
  <tr>
    <td width="180" valign="middle">
      <img src="olidra.png" width="160" alt="Olidra">
    </td>
    <td valign="middle">
      <p>Motor de xadrez UCI desenvolvido em Rust, que combina busca MCTS com PUCT e múltiplas threads em uma árvore compartilhada. Utiliza duas redes neurais compactas, no estilo NNUE, para estimar posições (<i>value</i>) e orientar a seleção de lances (<i>policy</i>), com inferência quantizada em int8 na CPU.</p>
      <p>As redes foram treinadas com aproximadamente 205 milhões de posições do banco de avaliações do Lichess, balanceadas entre abertura, meio-jogo e final. A rede de <i>value</i> aprende com avaliações do Stockfish, enquanto a de <i>policy</i> combina a destilação dos oito principais lances do Lc0 (peso 0,7) com o melhor lance do Stockfish (peso 0,3).</p>
    </td>
  </tr>
</table>

## Compilar
```shell
cargo build --release
```

O executável fica em `target/release/olidra.exe` e procura as redes em `nets/` (na pasta do executável ou na raiz do projeto). Para as redes usarem AVX2, compile para a CPU da máquina:

```shell
RUSTFLAGS="-C target-cpu=native" cargo build --release
cargo test --release
```

## Opções UCI

| opção | padrão | |
|---|---|---|
| Hash | 1024 | MB da árvore (nós e arestas); cheia, ela é compactada |
| Threads | 1 | threads descendo a mesma árvore (virtual loss) |
| CPuct, CPuctBase, CPuctFactor | 175, 38739, 389 | cpuct = CPuct + CPuctFactor · ln((N + base) / base), /100 |
| MoveOverhead | 30 | ms descontados de cada lance (latência da GUI ou da rede) |
| EvalFile, PolicyFile | nets/value.bin, nets/policy.bin | redes |
| MultiPV | 1 | linhas mostradas (os lances mais visitados da raiz) |
| InstrumentalProfile | false | imprime no fim da busca o perfil instrumental: visitas, priors e valores de cada nó da variante principal |
| Findings | false | imprime no fim da busca os lances contraintuitivos achados na árvore (prior baixo, maioria das visitas) |
