{
  description = "A small demonstration of an anti-sleeper-agent technique";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.poetry2nix = {
    url = "github:nix-community/poetry2nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { system = "x86_64-linux"; };
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication mkPoetryEnv defaultPoetryOverrides;
        myOverrides = [
          (final: prev:
            {
              equinox = prev.equinox.overridePythonAttrs (old: {
                buildInputs = (old.buildInputs or [ ]) ++ [ final.hatchling ];
              });
            })
          (defaultPoetryOverrides.overrideOverlay (final: prev: {
            # `rich` has some dependencies that are problematic at build time
            # (they bring in the all-but-abandoned `future` which is incompatible with python 3.12)
            # but overriding the override like this removes the `commonmark` dependency
            # specified in `defaultPoetryOverrides`
            rich = prev.rich.overridePythonAttrs (old: { });
          }))
        ];
        poetryAttrs = (extras: {
          projectDir = ./.;
          preferWheels = true;
          python = pkgs.python311;
          overrides = myOverrides;
          extras = extras;
        });
      in
      rec {
        formatter = pkgs.nixpkgs-fmt;
        # Just a dummy env to because a default package is required
        packages.default = pkgs.python311;
        devShells.default = (mkPoetryEnv (poetryAttrs [ "dev" "vscode" ])).env.overrideAttrs (final: prev: {
          # pip required for VSCode interactive window
          # We maybe should specify `ruff` in the pyproject.toml but then `poetry2nix` downloads the whole rust
          # toolchain to build it from source
          nativeBuildInputs = (prev.nativeBuildInputs or [ ]) ++ [ pkgs.poetry pkgs.ruff ];
          shellHook = (prev.shellHook or "") + ''
            export PATH=node_modules/.bin/:$PATH
          '';
        });
      }
    );
}
