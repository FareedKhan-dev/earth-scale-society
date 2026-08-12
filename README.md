# earth-scale-society

A social simulation of **one billion agents** that makes **zero language-model calls at runtime**.

Agent-based models have simulated millions of people for decades, on hand-written rules. Language
models finally gave those agents realistic behaviour, and capped the population at a few thousand,
because every interaction is an inference call. This repository is the other way round: a large open
model is paid **once, offline**, to answer every question the simulation can ever ask, the answers are
frozen into a lookup table, and the billion-agent run reads that table instead of a model.

The whole build is one notebook, [`earth-scale-society.ipynb`](earth-scale-society.ipynb), ninety two
code cells top to bottom.

## Why it works

A billion agents drawn from a pool of 10,000 personas, each holding one of three stances, can only ever
ask

```
10,000 speakers x 10,000 listeners x 3 x 3 = 900,000,000 distinct questions
```

and **that number does not grow with the population**. So it is enumerable. We distil a 235B teacher
into a 4.47M-parameter surrogate, sweep the surrogate over all 900M keys, and pack the result into one
byte per entry.

## Results

| | |
|---|---|
| Population | 1,000,000,000 agents |
| Graph | 2,999,999,994 edges built, 1,493,000,000 kept after the influencer prune |
| Lookup table | 900,000,000 entries, 858.31 MiB, one byte each |
| Resident memory | 11.197 GB, or 9.772 bytes per agent |
| One round | 14,930,000 interactions in 1.04 s on 32 cores |
| One hundred rounds | 104.2 s, **0 model calls** |
| Teacher | Qwen3-235B-A22B-Instruct-2507-FP8, 270,000 questions in 2h 59m |
| Surrogate | 4,467,203 parameters, 52,606x smaller than its teacher |
| End to end | 4h 10m on 4x H100, of which the billion-agent run is 0.69% |

The table stores a **quantised posterior**, not an argmax. Storing the winning class instead would cost
the same byte and destroy 18 of the 27 transition channels, freezing every stance into an absorbing
state so nobody ever changes their mind again.

## Hardware

Built and measured on **4x NVIDIA H100 80GB** with 64 CPU cores and 128 GB of host RAM.

Only the teacher needs all four GPUs. The billion-agent runtime is a NumPy program that fits in
11.2 GB, so it runs on one GPU with room to spare, or on CPU. Peak host memory during the graph build
is 34 GB.

## Install

```bash
git clone https://github.com/FareedKhan-dev/earth-scale-society.git
cd earth-scale-society
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.12.

## Getting the data

This project uses **Wave 7 of the World Values Survey** (2017 to 2022, 64 countries) as its persona
corpus. It is free, but it is not redistributable and it is not anonymous to download, so it is **not
in this repository** and never will be.

1. Register at <https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp>
2. Accept the research-use terms and request the cross-national CSV
3. Put the file at `data/WVS_Cross-National_Wave_7_csv_v6_0.csv`

There is no direct URL, so this cannot be a `wget` you paste.

A second reason it is not here: a persona in this project is a near-verbatim rendering of one real
respondent's full record, which is re-identifying for rare strata. Code and embeddings are shareable.
Personas joined back to source rows are not, and the notebook never writes them.

`data/wvs7_codebook.json` **is** in the repository. It maps the numeric response codes to labels for
the 26 variables this project reads. It is our transcription for those variables only, so check it
against the official WVS variable documentation before trusting it for anything else.

## Run it

The teacher runs in its own shell, because the notebook talks to it over HTTP:

```bash
bash serve_teacher.sh
```

Then the notebook, top to bottom. There is no timeout because the graph build takes 55 minutes and the
teacher pass takes three hours:

```bash
jupyter nbconvert --to notebook --execute earth-scale-society.ipynb \
  --output runs/latest.ipynb \
  --ExecutePreprocessor.timeout=-1
```

Scale is one knob. `configs/billion.yaml` is the flagship run; `configs/small.yaml` is 20,000 agents,
small enough to keep a live model in the loop for the fidelity comparison; `configs/long.yaml` is 5,000
rounds, which is where the equilibrium claims come from.

## Layout

```
earth-scale-society/
├── earth-scale-society.ipynb   # the whole build, 92 code cells, 28 figures
├── serve_teacher.sh            # vLLM, Qwen3-235B-A22B-Instruct-2507-FP8, TP=4
├── requirements.txt
├── configs/
│   ├── billion.yaml            # the flagship run, one billion agents
│   ├── small.yaml              # 20,000 agents, small enough for a live model
│   └── long.yaml               # 5,000 rounds, for the equilibrium claims
└── data/
    └── wvs7_codebook.json      # survey codes to labels, the one data file we ship
```

Everything else is written when you run it, and is git-ignored:

| Path | Written by | Size |
|---|---|---|
| `data/survey_wave.jsonl` | Census cells | 96,125 records |
| `artifacts/pool_embeddings.npy` | Census cells | 39.06 MiB |
| `artifacts/lattice/` | Lattice cells | 6.772 GB, memory-mapped shards |
| `artifacts/atlas.npy` | Atlas cells | 858.31 MiB |
| `ckpt/echo_ep24.pt` | Echo cells | 17.87 MB |
| `runs/latest.ipynb` | `nbconvert` | the executed notebook |

`artifacts/atlas.npy` is past GitHub's 100 MB per-file limit, so it could not be committed even if it
were useful to.

## What this does not show

The runtime is not a language model. It is a stochastic automaton whose transition rule a language
model estimated once. Ten thousand personas stand in for a billion people. There is no memory and no
message content, so repeated exposure is memoryless. Every fidelity number measures agreement with one
Qwen model, not with reality. A worked example in one prompt moved trust further than the entire social
class gradient, so absolute levels are prompt-relative and only the orderings survive.

The billion demonstrates the machinery. Above a million agents the terminal answer is flat to three
decimal places, so the billion is not required by the science.

## Prior work

The idea of a billion-agent social simulation comes from
[Modeling Earth-Scale Human-Like Societies with One Billion Agents](https://arxiv.org/abs/2506.12078).
That work ran on closed models. This one reaches the same scale on open weights and rentable hardware,
and publishes every prompt it uses.

## License

MIT. See [LICENSE](LICENSE).

The World Values Survey data is **not** covered by this licence and is not distributed here. It remains
subject to the WVS terms you accept when you download it.
