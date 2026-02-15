# Mega IA

Projeto focado em arquitetura, governança de modelos e operação segura de API em domínio de baixo sinal estatístico.

Sistema full-stack para geração e ranqueamento de combinações de Mega-Sena, Quina e Lotofácil a partir de dados históricos e modelo de Machine Learning.

- análise estatística de frequência, repetição e pares;
- geração aleatória de candidatos;
- ranking por modelo de ML e regras de diversidade/cobertura.

O score retornado é **relativo ao ranking da rodada** e **não representa probabilidade real de acerto**.

## O que o projeto faz

1. Carrega dados históricos por modalidade (`datasets/mega.csv`, `datasets/quina.csv`, `datasets/lotofacil.csv`).
2. Extrai contexto estatístico (hot/warm/cold, pares, repetições).
3. Gera milhares de jogos candidatos.
4. Ranqueia candidatos com modelo supervisionado treinado por modalidade.
5. Retorna os melhores jogos via API e dashboard.

## Stack

- Python 3.11
- FastAPI + Uvicorn
- Pandas + NumPy
- scikit-learn + skops
- Redis (opcional, rate limit e lock distribuído)

## Estrutura

```text
api/              # FastAPI, rotas, middlewares, rate limit, telemetria
core/             # domínio: dados, validações, geração, serviços
ml/               # treino, inferência, métricas e artefatos
dashboard/        # frontend estático (index + app.jsx + style.css)
datasets/         # CSVs históricos por modalidade
tests/            # testes unitários/integração
scripts/          # utilitários (ex.: A/B de ranking)
ARCHITECTURE.md   # visão técnica da arquitetura
```

## Instalação

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Execução local

Treinar modelos:

```bash
python main.py --train --modality mega
python main.py --train --modality quina
python main.py --train --modality lotofacil
```

Subir API + dashboard:

```bash
python main.py --serve
```

Acessos:

- Dashboard: `http://localhost:8000`
- Health: `http://localhost:8000/healthz`
- Readiness: `http://localhost:8000/readyz`

## Execução com Docker

```bash
docker compose up --build
```

## Uso via CLI

Gerar jogos:

```bash
python main.py --games 10 --modality mega --candidates 5000
```

## API principal

- `GET /generate/{lottery}?n_games=10&candidates=5000&save_csv=false`
- `POST /models/train/{lottery}`
- `GET /models/train/status/{job_id}`
- `GET /models/info/{lottery}`
- `GET /models/supported`
- `GET /metrics`
- `GET /healthz`
- `GET /readyz?lottery=mega`

Regras:

- limite de `n_games`: 100;
- limite de candidatos e custo por request via configuração;
- score de jogo: sinal relativo para ordenação.

## Variáveis de ambiente

Use `.env.example` como base.

Principais grupos:

- segurança: API keys e HMAC de artefatos;
- rate limit: backend `memory`/`redis` e limites de janela;
- treino: cooldown, retry, fila e gates;
- API/UI: CORS, métricas e proxy headers.

## Datasets

Arquivos esperados:

- `datasets/mega.csv` com colunas `n1..n6`
- `datasets/quina.csv` com colunas `n1..n5`
- `datasets/lotofacil.csv` com colunas `n1..n15`

Observação:

- mantenha delimitador consistente no arquivo inteiro.

## Testes e qualidade

```bash
python -m unittest discover -s tests -p "test_*.py"
python -m black --check .
python -m isort --check-only .
python -m ruff check .
```

## Troubleshooting rápido

- `Modelo nao encontrado`: rode treino da modalidade.
- `Artefato invalido para skops`: limpe artefatos da modalidade e retreine.
- `Nenhuma linha valida encontrada`: valide delimitador e schema do CSV.

## Próximos passos

- Refatorar train.py por responsabilidade.
- Refatorar training_jobs.py por responsabilidade.
- Melhorar performance da inferência para candidates altos.
- Fortalecer validação estatística por modalidade.
- Adicionar teste de carga básico do endpoint /generate/{lottery}.

## Documentação técnica

- Arquitetura: `ARCHITECTURE.md`

## Autor
Erick Ribeiro Moreira  
Backend Developer — Python, Applied AI, Intelligent Systems.