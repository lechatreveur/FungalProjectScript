#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 12:35:56 2025

@author: user
"""

import os, shutil, subprocess, shlex

def resolve_ffmpeg():
    p = shutil.which("ffmpeg")
    if p: return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise FileNotFoundError("ffmpeg not found; install system ffmpeg or `pip install imageio-ffmpeg`") from e

def concat_videos(out_path, inputs, width=1280, fps=5, crf=18):
    ffmpeg = resolve_ffmpeg()
    missing = [p for p in inputs if not os.path.isfile(p)]
    if missing: raise FileNotFoundError(f"Missing inputs: {missing}")
    vf_parts = [f"[{i}:v]scale={width}:-2,fps={fps},format=yuv420p[v{i}]" for i in range(len(inputs))]
    vf = ";".join(vf_parts) + ";" + "".join(f"[v{i}]" for i in range(len(inputs))) + f"concat=n={len(inputs)}:v=1:a=0[v]"
    cmd = [ffmpeg]
    for p in inputs: cmd += ["-i", p]
    cmd += ["-filter_complex", vf, "-map", "[v]", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", out_path]
    subprocess.run(cmd, check=True)
    return out_path
