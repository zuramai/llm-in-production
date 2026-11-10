# Chapter N: <Title>

> **Tested on:** YYYY-MM-DD · Ubuntu 22.04 (NVIDIA RTX 3090, 24 GB) · Python 3.12 ·
> <models + providers> · <key library versions>

<Opening hook: a concrete scene from a real team, then what is happening in the industry right
now, dated and sourced.>

![<alt>](images/NN-<slug>.svg)

## The problem

<Who has the problem, what they do today, why it hurts. One example interaction.>

## Requirements

| Type | Requirement | Target |
|---|---|---|
| Functional | ... | ... |
| Non-functional | p95 latency | < N s |
| Non-functional | cost | < $N per 1k requests |
| Non-functional | accuracy | ≥ N% on the eval set |

## The data

<Mermaid erDiagram of the tables used, the seed commands, and a few sample rows from a real query.>

## Concepts

<Theory needed for this chapter, in plain language, each idea with a visual.>

> [!NOTE]
> **In the wild:** ... ([source](https://...))

## v0: the naive solution

<Simplest thing that could work. Code excerpt from a tested file.>

## Measuring v0

<Real outputs, real failures. Table.>

## Iterating

### 1. <Improvement>

<What's wrong → options → decision → code → measured effect.>

> [!IMPORTANT]
> **Decision: X over Y** — <why; when you'd choose Y instead>.

### 2. <Improvement>

## Production hardening

> [!WARNING]
> <Pitfall.>

## Serving it

<How to run it as a service. Screenshot.>

## Results

| Variant | Accuracy | p50 latency | p95 latency | $ / 1k requests |
|---|---|---|---|---|

## Exercises

1. ...
2. ...

## What we learned

- ...

## References

- <Title> — <Publisher>, <date>. <URL>
