#!/usr/bin/env bash
# Manage git worktrees at .worktrees/<branch>.
#
# Usage:
#   scripts/worktree.sh add  <branch> [--tool claude|codex|none]
#   scripts/worktree.sh rm   <branch> [--force] [--keep-branch]
#   scripts/worktree.sh list
#
# Shorthand: `scripts/worktree.sh <branch> [...]` is equivalent to `add <branch>`.

set -euo pipefail

usage() {
	cat <<'EOF'
Usage:
  scripts/worktree.sh add  <branch> [--tool claude|codex|none]
  scripts/worktree.sh rm   <branch> [--force] [--keep-branch]
  scripts/worktree.sh list
  scripts/worktree.sh <branch> [--tool ...]    # shorthand for `add`

add:
  --tool claude|codex|none   Tool to launch in the worktree. Default: claude.
                             "none" only prints the path and exits.

rm:
  --force                    Skip dirty/unpushed safety checks.
  --keep-branch              Keep the branch after removing the worktree.

  -h, --help                 Show this help.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
warn() { printf 'warning: %s\n' "$*" >&2; }
info() { printf '==> %s\n' "$*"; }

resolve_paths() {
	local branch="$1"
	git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not inside a git working tree"
	GIT_COMMON_DIR=$(cd "$(git rev-parse --git-common-dir)" && pwd -P)
	MAIN_REPO=$(dirname "$GIT_COMMON_DIR")
	WORKTREE_REL=".worktrees/$branch"
	WORKTREE_ABS="$MAIN_REPO/$WORKTREE_REL"
	WT_GITDIR="$GIT_COMMON_DIR/worktrees/$branch"
}

cmd_add() {
	local branch="" tool="claude"
	while (($#)); do
		case "$1" in
			--tool)
				[[ $# -ge 2 ]] || die "--tool requires an argument"
				tool="$2"; shift 2 ;;
			--tool=*) tool="${1#--tool=}"; shift ;;
			--) shift; break ;;
			-*) die "unknown flag: $1" ;;
			*)
				[[ -z "$branch" ]] || die "unexpected positional arg: $1"
				branch="$1"; shift ;;
		esac
	done
	[[ -n "$branch" ]] || { usage >&2; exit 2; }
	case "$tool" in claude|codex|none) ;; *) die "--tool must be claude, codex, or none" ;; esac

	resolve_paths "$branch"

	[[ -e "$WORKTREE_ABS" ]] && die "$WORKTREE_REL already exists"

	local gitignore="$MAIN_REPO/.gitignore"
	if ! grep -qxF '/.worktrees/' "$gitignore" 2>/dev/null; then
		info "appending /.worktrees/ to .gitignore"
		printf '/.worktrees/\n' >> "$gitignore"
		warn ".gitignore was updated — remember to commit it"
	fi

	if [[ -n "$(git -C "$MAIN_REPO" status --porcelain)" ]]; then
		warn "main repo has uncommitted changes; the new worktree starts from current HEAD"
	fi

	info "creating worktree at $WORKTREE_REL"
	if git -C "$MAIN_REPO" show-ref --verify --quiet "refs/heads/$branch"; then
		git -C "$MAIN_REPO" worktree add "$WORKTREE_REL" "$branch"
	else
		git -C "$MAIN_REPO" worktree add -b "$branch" "$WORKTREE_REL"
	fi

	info "ready: $WORKTREE_ABS"

	if [[ "$tool" == "none" ]]; then
		exit 0
	fi

	command -v "$tool" >/dev/null 2>&1 || die "$tool not found on PATH"

	cd "$WORKTREE_ABS"
	"$tool" || warn "$tool exited with status $?"
	exec "${SHELL:-/bin/bash}"
}

cmd_rm() {
	local branch="" force=0 keep_branch=0
	while (($#)); do
		case "$1" in
			--force) force=1; shift ;;
			--keep-branch) keep_branch=1; shift ;;
			--) shift; break ;;
			-*) die "unknown flag: $1" ;;
			*)
				[[ -z "$branch" ]] || die "unexpected positional arg: $1"
				branch="$1"; shift ;;
		esac
	done
	[[ -n "$branch" ]] || { usage >&2; exit 2; }

	resolve_paths "$branch"

	[[ -d "$WORKTREE_ABS" ]] || die "$WORKTREE_REL does not exist"

	if (( ! force )); then
		local dirty
		dirty=$(git -C "$WORKTREE_ABS" status --porcelain 2>/dev/null || true)
		[[ -z "$dirty" ]] || die "$WORKTREE_REL has uncommitted changes; pass --force to discard"

		if git -C "$MAIN_REPO" show-ref --verify --quiet "refs/heads/$branch"; then
			local upstream
			upstream=$(git -C "$MAIN_REPO" rev-parse --abbrev-ref "$branch@{upstream}" 2>/dev/null || true)
			if [[ -n "$upstream" ]]; then
				local ahead
				ahead=$(git -C "$MAIN_REPO" rev-list --count "$upstream..$branch" 2>/dev/null || echo 0)
				[[ "$ahead" == "0" ]] || die "branch $branch has $ahead unpushed commit(s); pass --force to discard"
			else
				# No upstream — safe iff the tip is contained in some other ref.
				local tip reachable_from
				tip=$(git -C "$MAIN_REPO" rev-parse "refs/heads/$branch")
				reachable_from=$(git -C "$MAIN_REPO" for-each-ref --contains "$tip" \
					--format='%(refname)' refs/heads refs/remotes \
					| grep -vxF "refs/heads/$branch" | head -n1)
				[[ -n "$reachable_from" ]] || die "branch $branch has no upstream and commits not reachable from any other ref; pass --force to discard"
			fi
		fi
	fi

	info "removing worktree $WORKTREE_REL"
	rm -rf "$WORKTREE_ABS"

	# Drop git's internal worktree metadata if the directory was removed directly.
	if [[ -d "$WT_GITDIR" ]]; then
		info "removing git worktree metadata $WT_GITDIR"
		rm -rf "$WT_GITDIR"
	fi
	git -C "$MAIN_REPO" worktree prune

	if (( keep_branch )); then
		info "keeping branch $branch"
	elif git -C "$MAIN_REPO" show-ref --verify --quiet "refs/heads/$branch"; then
		info "deleting branch $branch"
		git -C "$MAIN_REPO" branch -D "$branch"
	fi

	# Clean up empty .worktrees/ to keep `git status` tidy.
	rmdir "$MAIN_REPO/.worktrees" 2>/dev/null || true

	info "removed."
}

cmd_list() {
	(($# == 0)) || die "list takes no arguments"
	git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not inside a git working tree"
	git worktree list
}

case "${1:-}" in
	-h|--help) usage; exit 0 ;;
	"") usage >&2; exit 2 ;;
	add) shift; cmd_add "$@" ;;
	rm|remove) shift; cmd_rm "$@" ;;
	list|ls) shift; cmd_list "$@" ;;
	-*) die "unknown flag: $1" ;;
	*) cmd_add "$@" ;;  # implicit `add <branch>`
esac
