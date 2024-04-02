# Unconditional conditioning: Removing sleeper agent behavior

This repo contains code for a technique that attempts to remove [sleeper agent](https://arxiv.org/abs/2401.05566) behavior. There's an accompanying [blog post](https://col-ex.org/posts/sleeper-agent/).

## Usage

As far as dependencies and setup, there are a couple of options:

- There's a `pyproject.toml` that contains Poetry declarations, but I never use `poetry` directly so I can't guarantee its correctness.
- I use Poetry indirectly via `poetry2nix`. If you're a Nix user, invoking `nix develop` should get you a CPU-only setup.
