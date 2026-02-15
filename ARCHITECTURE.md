# Architecture

## Visão geral

Aplicação monolítica modular em Python com quatro blocos:

- `api/`: camada HTTP, middlewares, rate limit e telemetria.
- `core/`: regras de domínio, carga de dados e geração.
- `ml/`: treino, inferência e governança de artefatos.
- `dashboard/`: frontend estático servido pela API.

## Fluxo de geração

1. Cliente chama `GET /generate/{lottery}` com `n_games` e `candidates`.
2. API aplica auth, rate limit e validação de custo.
3. `core.service.generate_ranked_games`:
   - gera candidatos aleatórios;
   - carrega artefato promovido da modalidade;
   - ranqueia e retorna top-N.
4. API responde jogos formatados e score relativo.

## Fluxo de treino

1. Cliente chama `POST /models/train/{lottery}`.
2. Job entra no `TrainingJobManager` (fila assíncrona).
3. `ml.train.train_lottery_model` executa:
   - montagem de dataset temporal;
   - treino + calibração;
   - backtest walk-forward;
   - gate de promoção.
4. Artefato promovido é persistido em `ml/artifacts/`.

## Componentes principais

- `api/app.py`: bootstrap da aplicação.
- `api/routes.py`: contratos HTTP.
- `api/middlewares/request_context.py`: request-id, log estruturado e latência.
- `api/middlewares/rate_limit_middleware.py`: throttling por escopo.
- `api/rate_limit_backend.py`: backends `memory` e `redis`.
- `core/loader.py`: normalização e validação de CSV.
- `core/service.py`: orquestração de geração ranqueada.
- `ml/train.py`: ciclo de vida de modelo e promoção.
- `ml/infer.py`: score/ranking em runtime.

## Dados e artefatos

- Entrada: `datasets/mega.csv`, `datasets/quina.csv`, `datasets/lotofacil.csv`.
- Saída opcional: `output/*.csv`.
- Artefato: `.skops` + metadata JSON + hash + assinatura HMAC (quando habilitada).

## Segurança e operação

- API keys separadas por escopo (`operacao`, `treino`, `info`, `metricas`).
- Rate limit separado para `generate` e `train`.
- Endpoints de saúde: `healthz` e `readyz`.
- Integridade de artefato validada no load.

## Limite estatístico

O sistema não prediz sorteio real.

O score exposto é um **sinal relativo de ranking** entre candidatos da mesma rodada,
usado para priorização de jogos gerados.

## Decisões arquiteturais

- Monólito modular para reduzir complexidade operacional.
- Treino assíncrono para não bloquear requests HTTP.
- Rate limit por escopo para evitar bloqueio cruzado entre geração e treino.
- Promoção de artefatos condicionada à estabilidade temporal.

## Separação de responsabilidades

- A API não contém lógica de domínio.
- O core não depende de implementação concreta de ML.
- O módulo de ML não conhece detalhes de transporte HTTP.
