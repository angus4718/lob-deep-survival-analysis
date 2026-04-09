"""Notebook setup helpers for environment and package bootstrapping."""

from __future__ import annotations

import glob
import importlib.util
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import zipfile


def run(cmd, env=None):
    print(">", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def cleanup_stale_torch(pkg_dir: str) -> None:
    patterns = ["torch", "torch-*", "functorch", "functorch-*"]
    removed = False
    for pattern in patterns:
        for path in glob.glob(os.path.join(pkg_dir, pattern)):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
                removed = True
            except Exception:
                pass
    if removed:
        print("Removed stale torch/functorch artifacts from PKG_DIR.")


def gcc_major(gpp_path: str) -> int:
    try:
        out = subprocess.run([gpp_path, "--version"], capture_output=True, text=True, timeout=5).stdout
        match = re.search(r"(\d+)\.\d+", out)
        return int(match.group(1)) if match else -1
    except Exception:
        return -1


def compiler_works(gpp_path: str) -> bool:
    if not gpp_path or not os.path.exists(gpp_path):
        return False
    try:
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "test.cpp")
            out = os.path.join(td, "test.out")
            with open(src, "w", encoding="utf-8") as f:
                f.write("int main(){return 0;}\n")
            proc = subprocess.run([gpp_path, src, "-o", out], capture_output=True, text=True, timeout=10)
            return proc.returncode == 0
    except Exception:
        return False


def wheel_file_valid(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        return False
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            return any(name.endswith(".dist-info/WHEEL") for name in names)
    except Exception:
        return False


def wheelhouse_has(wheel_dir: str, dist_name: str) -> bool:
    want = dist_name.lower().replace("-", "_")
    for whl in glob.glob(os.path.join(wheel_dir, "*.whl")):
        stem = os.path.basename(whl)[:-4]
        dist = stem.split("-")[0].lower().replace("-", "_")
        if dist == want and wheel_file_valid(whl):
            return True
    return False


def purge_invalid_wheels(wheel_dir: str, dist_name: str) -> None:
    want = dist_name.lower().replace("-", "_")
    for whl in glob.glob(os.path.join(wheel_dir, "*.whl")):
        stem = os.path.basename(whl)[:-4]
        dist = stem.split("-")[0].lower().replace("-", "_")
        if dist == want and not wheel_file_valid(whl):
            try:
                os.remove(whl)
                print(f"Removed invalid wheel: {whl}")
            except Exception:
                pass


def load_module_env(module_name: str) -> bool:
    cmd = f"source /etc/profile.d/modules.sh && module load {module_name} >/dev/null 2>&1 && env -0"
    try:
        proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, timeout=20)
        if proc.returncode != 0:
            return False
        env_blob = proc.stdout.decode(errors="ignore")
        for item in env_blob.split("\x00"):
            if not item or "=" not in item:
                continue
            key, value = item.split("=", 1)
            if key in {
                "PATH",
                "LD_LIBRARY_PATH",
                "LIBRARY_PATH",
                "CPATH",
                "C_INCLUDE_PATH",
                "CPLUS_INCLUDE_PATH",
                "PKG_CONFIG_PATH",
                "CUDA_HOME",
                "CUDA_PATH",
            }:
                os.environ[key] = value
        return True
    except Exception:
        return False


def download_file_insecure(url: str, dst: str) -> bool:
    try:
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx, timeout=30) as resp, open(dst, "wb") as out:
            out.write(resp.read())
        return True
    except Exception:
        pass

    curl = shutil.which("curl")
    if curl:
        proc = subprocess.run([curl, "-L", "-k", "--fail", "-o", dst, url], capture_output=True, text=True)
        if proc.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
            return True

    wget = shutil.which("wget")
    if wget:
        proc = subprocess.run([wget, "--no-check-certificate", "-O", dst, url], capture_output=True, text=True)
        if proc.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
            return True

    return False


def try_download_causal_wheel(wheel_dir: str) -> bool:
    if wheelhouse_has(wheel_dir, "causal-conv1d"):
        return True

    try:
        import torch

        base = torch.__version__.split("+")[0]
        parts = base.split(".")
        version_tags = [".".join(parts[:2]), base] if len(parts) >= 2 else [base]
        abi_tag = "TRUE" if bool(torch._C._GLIBCXX_USE_CXX11_ABI) else "FALSE"
        abi_tags = [abi_tag, "FALSE" if abi_tag == "TRUE" else "TRUE"]
    except Exception:
        version_tags = []
        abi_tags = ["TRUE", "FALSE"]

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    for tv in version_tags:
        for abi in abi_tags:
            wheel_name = f"causal_conv1d-1.6.1+cu12torch{tv}cxx11abi{abi}-{py_tag}-{py_tag}-linux_x86_64.whl"
            url = f"https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1/{wheel_name}"
            dst = os.path.join(wheel_dir, wheel_name)
            print("Trying causal-conv1d wheel:", url)
            if download_file_insecure(url, dst):
                if wheel_file_valid(dst):
                    print("Downloaded valid wheel:", dst)
                    return True
                try:
                    os.remove(dst)
                except Exception:
                    pass

    print("Direct causal-conv1d wheel download failed for all URL variants.")
    return False