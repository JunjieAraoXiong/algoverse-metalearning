---
description: Commit staged changes, push branch, and open a PR
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git branch:*), Bash(git commit:*), Bash(git push:*), Bash(gh pr create:*)
---

## Context

- Current git status:
  !`git status --short`

- Current branch:
  !`git branch --show-current`

- Staged diff:
  !`git diff --cached --stat`

- Recent commits:
  !`git log --oneline -5`

## Task

1. If there are no staged changes, stop and explain why.
2. Write a concise, conventional commit message.
3. Commit **only staged changes**.
4. Push to the current branch.
5. Open a pull request with:
   - Clear title
   - Short description of what changed and why
