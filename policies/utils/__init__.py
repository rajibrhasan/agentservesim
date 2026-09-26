"""Helpers the policies need, and tooling that reads what they produced.

Nothing here decides anything, which is the test for belonging: `waste_model`
is arithmetic InferCept's rule evaluates, `kv_control` is the interface a
decision is applied through, `parity` diffs two decision logs after the fact.
A file that makes a choice about a program belongs one level up, in the module
of the paper that makes it.
"""
