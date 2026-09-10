"""Autonomous task orchestrator for the Geometric Consistency Probe.

Supervises Tasks 6-18 (see research/RESEARCH_PLAN.md): reads
state/progress.json, determines the next incomplete task, invokes
Claude Code non-interactively to implement it, independently verifies
the result (orchestrator/qa.py -- never trusting Claude's own
self-report, per research/RESEARCH_INVARIANTS.md invariant 18), and
either checkpoints the work with a git commit and advances, or feeds a
fix prompt back to Claude and retries up to a configurable limit.

See orchestrator/run.py for the CLI entry point.
"""
