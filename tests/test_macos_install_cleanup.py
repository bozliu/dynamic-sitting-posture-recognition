from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _create_fake_app(root: Path, name: str = "PostureMirror.app") -> Path:
    app = root / name
    exec_path = app / "Contents" / "MacOS" / "PostureMirror"
    exec_path.parent.mkdir(parents=True, exist_ok=True)
    exec_path.write_text("#!/bin/sh\necho posturemirror\n", encoding="utf-8")
    exec_path.chmod(0o755)
    return app


def _run(script: Path, args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, env=env, check=True)


def test_install_overwrite_has_no_backup_dirs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "install_macos_app.sh"

    apps_dir = tmp_path / "Applications"
    apps_dir.mkdir()
    old_app = _create_fake_app(apps_dir)
    (old_app / "old_marker.txt").write_text("old", encoding="utf-8")

    source_root = tmp_path / "source"
    source_root.mkdir()
    source_app = _create_fake_app(source_root)
    (source_app / "new_marker.txt").write_text("new", encoding="utf-8")

    _run(script, ["--source-app", str(source_app), "--apps-dir", str(apps_dir)])

    target = apps_dir / "PostureMirror.app"
    assert target.exists()
    assert (target / "new_marker.txt").exists()
    assert not (target / "old_marker.txt").exists()
    assert list(apps_dir.glob("PostureMirror.app.bak.*")) == []
    assert list(apps_dir.glob("PostureMirror.app.old*")) == []


def test_cleanup_only_removes_whitelisted_residuals(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "cleanup_app_residuals.sh"

    apps_dir = tmp_path / "Applications"
    apps_dir.mkdir()

    _create_fake_app(apps_dir)
    (apps_dir / "PostureMirror.app.bak.1001").mkdir()
    (apps_dir / "PostureMirror.app.old-1").mkdir()
    (apps_dir / "Other.app.bak.1").mkdir()

    _run(script, ["--apps-dir", str(apps_dir)])

    assert (apps_dir / "PostureMirror.app").exists()
    assert not (apps_dir / "PostureMirror.app.bak.1001").exists()
    assert not (apps_dir / "PostureMirror.app.old-1").exists()
    assert (apps_dir / "Other.app.bak.1").exists()


def test_cleanup_dry_run_does_not_delete(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "cleanup_app_residuals.sh"

    apps_dir = tmp_path / "Applications"
    apps_dir.mkdir()
    residual = apps_dir / "PostureMirror.app.bak.1002"
    residual.mkdir()

    _run(script, ["--apps-dir", str(apps_dir), "--dry-run"])

    assert residual.exists()


def test_uninstall_default_keeps_user_data(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "uninstall_macos_app.sh"

    apps_dir = tmp_path / "Applications"
    apps_dir.mkdir()
    _create_fake_app(apps_dir)
    (apps_dir / "PostureMirror.app.bak.2001").mkdir()

    fake_home = tmp_path / "home"
    user_data = fake_home / "Library" / "Application Support" / "PostureMirror"
    user_data.mkdir(parents=True)

    env = dict(os.environ)
    env["HOME"] = str(fake_home)
    _run(script, ["--apps-dir", str(apps_dir)], env=env)

    assert not (apps_dir / "PostureMirror.app").exists()
    assert not (apps_dir / "PostureMirror.app.bak.2001").exists()
    assert user_data.exists()


def test_uninstall_with_purge_removes_user_data(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "uninstall_macos_app.sh"

    apps_dir = tmp_path / "Applications"
    apps_dir.mkdir()
    _create_fake_app(apps_dir)

    fake_home = tmp_path / "home"
    user_paths = [
        fake_home / "Library" / "Application Support" / "PostureMirror",
        fake_home / "Library" / "Caches" / "com.bozliu.posturemirror",
        fake_home / "Library" / "Preferences" / "com.bozliu.posturemirror.plist",
        fake_home / "Library" / "Saved Application State" / "com.bozliu.posturemirror.savedState",
    ]
    for path in user_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".plist":
            path.write_text("pref", encoding="utf-8")
        else:
            path.mkdir(exist_ok=True)

    env = dict(os.environ)
    env["HOME"] = str(fake_home)
    _run(script, ["--apps-dir", str(apps_dir), "--purge-user-data"], env=env)

    assert not (apps_dir / "PostureMirror.app").exists()
    for path in user_paths:
        assert not path.exists()
