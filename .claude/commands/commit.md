---
description: "Commit changes with a message beginning with 'Claude'"
allowed-tools: ["bash"]
---

# Commit Changes

Commit and begin message with Claude.

$ARGUMENTS

## Check Status and Commit
```bash
!git status
```

```bash
!git add -A
```

```bash
!git commit -m "$(cat <<'EOF'
Claude $ARGUMENTS

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

```bash
!git status
```