"""``deep-dream completion`` — generate shell completion scripts.

Uses Click 8's built-in ``click.shell_completion`` module to produce
completion scripts for Bash, Zsh, and Fish.

\b
Quick setup:
    # Bash — add to ~/.bashrc
    echo 'eval "$(deep-dream completion bash)"' >> ~/.bashrc

    # Zsh — save to a function directory
    deep-dream completion zsh > ~/.zfunc/_deep_dream

    # Fish — save to completions directory
    deep-dream completion fish > ~/.config/fish/completions/deep-dream.fish
"""
from __future__ import annotations

import click

from ._exit_codes import OK


@click.command()
@click.argument(
    "shell",
    type=click.Choice(["bash", "zsh", "fish"], case_sensitive=False),
)
@click.pass_context
def completion(ctx: click.Context, shell: str) -> int:
    """Generate shell completion scripts.

    Print a completion script for the given shell to stdout.  Source the
    output in your shell's init file to enable tab-completion for
    ``deep-dream`` commands, options, and arguments.

    \b
    Examples:
      # Bash
      echo 'eval "$(deep-dream completion bash)"' >> ~/.bashrc

      # Zsh
      deep-dream completion zsh > ~/.zfunc/_deep_dream

      # Fish
      deep-dream completion fish > ~/.config/fish/completions/deep-dream.fish
    """
    from core.cli._main import cli as root_cli

    import click.shell_completion

    shell_lower = shell.lower()

    # Map shell name to the corresponding Click shell-completion class.
    shell_cls = click.shell_completion.get_completion_class(shell_lower)
    if shell_cls is None:
        # Should not happen since Click's Choice validates input, but
        # handle gracefully just in case.
        click.echo(f"Error: no completion support for '{shell}'.", err=True)
        raise SystemExit(1)

    # Build the completer.  prog_name must match the command users type.
    prog_name = "deep-dream"
    complete_var = f"_{prog_name.replace('-', '_').upper()}_COMPLETE"
    completer = shell_cls(root_cli, {}, prog_name, complete_var)
    click.echo(completer.source())

    return OK
