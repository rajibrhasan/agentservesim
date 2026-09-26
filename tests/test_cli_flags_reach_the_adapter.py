"""Every `args.<name>` in the entry point must come from a defined flag.

A flag definition lost in an edit does not fail a unit test: the adapter takes
the keyword happily and nothing notices until a real run dies in argparse with
`Namespace has no attribute`. That is how --min-waste-fcfs-restore shipped
without its own flag on 2026-09-18, and it cost a three-run simulator sweep.
This is the cheap check that would have caught it.
"""
import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(ROOT, 'serving', '__main__.py')

# Attributes argparse or the entry point sets on the namespace itself.
SET_IN_CODE = {'run_id'}


def _tree():
    with open(MAIN) as f:
        return ast.parse(f.read())


def test_every_args_attribute_has_a_flag():
    tree = _tree()
    defined = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, 'attr', '') == 'add_argument'
                and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and node.args[0].value.startswith('--')):
            defined.add(node.args[0].value[2:].replace('-', '_'))
    used = {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name) and node.value.id == 'args'}
    missing = sorted(used - defined - SET_IN_CODE)
    assert not missing, f"args.{{{', '.join(missing)}}} has no --flag"


def test_the_policy_flags_added_for_the_paper_integrations_exist():
    """Named explicitly: an audit that only compares two derived sets passes
    just as happily when both are empty."""
    with open(MAIN) as f:
        src = f.read()
    for flag in ('--scheduling', '--routing',
                 '--autellix-service-boundaries', '--autellix-quanta',
                 '--autellix-starvation-ratio', '--autellix-overprovision',
                 '--autellix-swap', '--autellix-swap-bw', '--autellix-swap-profile',
                 '--long-prompt-tokens',
                 '--saga-eviction-order', '--saga-fairness',
                 '--saga-fairness-slack', '--saga-prefetch',
                 '--saga-prefetch-margin',
                 '--min-waste-swap', '--min-waste-swap-bw',
                 '--min-waste-fcfs-restore'):
        assert f"'{flag}'" in src, f"{flag} is not defined in serving/__main__.py"
