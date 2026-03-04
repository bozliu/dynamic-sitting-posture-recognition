# -*- mode: python ; coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

project_root = Path.cwd()
models_dir = Path(os.environ.get("POSTUREMIRROR_MODELS_DIR", str(project_root / "packaging" / "models")))

hiddenimports = []
hiddenimports += collect_submodules("posture_recognition")
hiddenimports += collect_submodules("ultralytics")

datas = []
datas += collect_data_files("posture_recognition")
if models_dir.exists():
    datas.append((str(models_dir), "models"))
datas.append((str(project_root / "configs"), "configs"))

block_cipher = None


a = Analysis(
    [str(project_root / "src/posture_recognition/gui/posturemirror_app.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PostureMirror",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="arm64",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="PostureMirror",
)

app = BUNDLE(
    coll,
    name="PostureMirror.app",
    icon=None,
    bundle_identifier="com.bozliu.posturemirror",
    version="2.0.1",
    info_plist={
        "CFBundleShortVersionString": "2.0.1",
        "CFBundleVersion": "2.0.1",
        "NSCameraUsageDescription": "PostureMirror uses the camera to provide real-time posture feedback.",
    },
)
