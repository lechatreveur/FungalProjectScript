#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:02:31 2024

@author: user
"""

import os
import subprocess
import pandas as pd
import xml.etree.ElementTree as ET

# Configuration
username = "ian.hsu54383@gmail.com"  # Replace with your JGI username
password = "Yen00789"                # Replace with your JGI password
cookies_file = "cookies"             # Cookies file path

excel_file = "/Users/user/Documents/FungalProject/FungalGenomesToPick.xlsx"  # Excel input file
#output_dir = "/Users/user/Documents/FungalProject/Genomes/"  # Directory to store downloaded files
output_dir = "/Volumes/Ian/FungalGenomes/"  # Directory to store downloaded files
os.makedirs(output_dir, exist_ok=True)

keywords = ["CDS", "proteins"]       # Keywords to search for in filenames

def login_to_jgi():
    """
    Logs in to JGI portal and saves session cookies.
    """
    print("Logging into JGI portal...")
    subprocess.run([
        "curl", "https://signon.jgi.doe.gov/signon/create",
        "--data-urlencode", f"login={username}",
        "--data-urlencode", f"password={password}",
        "-c", cookies_file
    ], check=True)
    print("Login successful! Cookies saved.")
    
def download_file(filename, file_url):
    """
    Download a file if it doesn't already exist.
    """
    file_url_full = f"https://genome.jgi.doe.gov{file_url.replace('&amp;', '&')}"
    output_file = os.path.join(output_dir, filename)

    if os.path.exists(output_file):
        print(f"File already exists, skipping: {filename}")
        return

    print(f"Downloading: {filename}")
    download_command = [
        "curl", "-L",
        file_url_full,
        "-b", cookies_file,
        "-o", output_file
    ]
    subprocess.run(download_command, check=True)
    print(f"Saved to: {output_file}")

def download_files_from_portal(portal_url):
    """
    Fetch files.xml from a portal URL and download latest files matching keywords.
    Only downloads the latest CDS and proteins (.aa.fasta.gz) files.
    """
    portal_name = portal_url.split("organism=")[-1]
    xml_output = os.path.join(output_dir, f"{portal_name}_files.xml")

    # Fetch files.xml
    print(f"Fetching files.xml for portal: {portal_name}...")
    fetch_command = [
        "curl", "-L",
        f"https://genome.jgi.doe.gov/portal/ext-api/downloads/get-directory?organism={portal_name}",
        "-b", cookies_file,
        "-o", xml_output
    ]
    result = subprocess.run(fetch_command, check=True, text=True, capture_output=True)

    if not os.path.exists(xml_output) or os.path.getsize(xml_output) == 0:
        print(f"Failed to fetch files.xml for {portal_name}. Response: {result.stdout}")
        return

    print(f"Parsing files.xml for portal: {portal_name}")
    tree = ET.parse(xml_output)
    root = tree.getroot()

    # Collect files by type
    cds_files = []
    protein_files = []

    for file in root.findall(".//file"):
        filename = file.attrib.get("filename", "")
        file_url = file.attrib.get("url", "")

        # Save only relevant CDS files
        if "CDS" in filename and filename.endswith(".fasta.gz"):
            cds_files.append((filename, file_url))

        # Save only protein sequence files (.aa.fasta.gz), not annotations
        elif "proteins" in filename and filename.endswith(".aa.fasta.gz"):
            protein_files.append((filename, file_url))

    def get_latest_file(files):
        """Return the (filename, url) of the latest file based on date in filename."""
        if not files:
            return None
        # Extract date from filename and sort
        files_sorted = sorted(
            files,
            key=lambda x: ''.join(filter(str.isdigit, x[0])),  # Extract digits to sort
            reverse=True
        )
        return files_sorted[0]  # Latest one

    # Download the latest CDS
    latest_cds = get_latest_file(cds_files)
    if latest_cds:
        filename, file_url = latest_cds
        download_file(filename, file_url)
    
    # Download the latest protein .aa.fasta.gz
    latest_protein = get_latest_file(protein_files)
    if latest_protein:
        filename, file_url = latest_protein
        download_file(filename, file_url)



def main():
    # Log in to JGI portal
    login_to_jgi()
    
    # Read the Excel file and process each portal
    print("Reading Excel file...")
    df = pd.read_excel(excel_file, sheet_name=0)
    print(df)
    for index, row in df.iterrows():
        portal_url = str(row.get("Web", "")).strip()
        if not portal_url or "https" not in portal_url:
            continue  # Skip invalid or empty portal links
        
        download_files_from_portal(portal_url)
    
    print("All downloads completed!")

if __name__ == "__main__":
    main()
