#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launch script for Fungal GUI.
Forces the correct Matplotlib backend since Spyder's internal environment is missing in standard terminals.
"""

import argparse
import os
import sys
import matplotlib

# Force TkAgg to bypass PySide/PyQt5 C++ linkage conflicts
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from SingleCellDataAnalysis.alignment_board_gui import review_septum_alignment_board_gui

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Fungal GUI for a specific experiment and film.")
    parser.add_argument("working_dir", nargs="?", default="/Volumes/X10 Pro/Movies/2025_12_31_M92", 
                        help="Root experiment directory")
    parser.add_argument("film_name", nargs="?", default=None,
                        help="Specific film field name (optional if discovery is used)")
    parser.add_argument("--filter", default="BF", 
                        help="Comma-separated substrings to filter films (e.g. 'BF' or '-2,-4,-6')")
    parser.add_argument("--mask_col", default="rle_bf", help="Mask column name (default: rle_bf)")
    parser.add_argument("--model", default=None, help="Path to model checkpoint")
    
    args = parser.parse_args()

    working_dir = args.working_dir
    film_input = args.film_name
    filter_strs = [s.strip() for s in args.filter.split(",") if s.strip()]

    # Multi-film discovery vs single film
    if film_input:
        films_to_load = [film_input]
    else:
        # Discover all subdirectories in working_dir
        if not os.path.isdir(working_dir):
            print(f"[ERROR] Directory not found: {working_dir}")
            sys.exit(1)
        
        all_dirs = [d for d in os.listdir(working_dir) if os.path.isdir(os.path.join(working_dir, d))]
        # Filter based on substrings
        films_to_load = []
        for d in sorted(all_dirs):
            # Check if any filter substring matches
            if any(fs in d for fs in filter_strs):
                films_to_load.append(d)
        
        if not films_to_load:
            print(f"[ERROR] No films found in {working_dir} matching filter: {args.filter}")
            sys.exit(1)

    print(f"Initializing GUI session for:")
    print(f"  Directory: {working_dir}")
    print(f"  Films ({len(films_to_load)}): {', '.join(films_to_load[:3])}{'...' if len(films_to_load) > 3 else ''}")
    print(f"  Filter:    {args.filter}")
    print("This may take 5-10 seconds...")
    
    # Run the GUI
    try:
        review_septum_alignment_board_gui(
            working_dir, 
            films_to_load, 
            block=False, 
            mask_col=args.mask_col,
            model_path=args.model
        )
        # Force the Matplotlib window event loop to stay open and take focus
        plt.show(block=True)
    except Exception as e:
        print(f"\n[ERROR] Failed to launch GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("GUI Closed.")
