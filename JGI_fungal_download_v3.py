#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 15:05:58 2025

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

excel_file = "/Users/user/Documents/Python_Scripts/FungalProjectScript/one_species_per_genus_with_manual_longest.csv"  # Excel input file
#output_dir = "/Users/user/Documents/FungalProject/Genomes/"  # Directory to store downloaded files
output_dir = "/Volumes/Ian/FungalGenomes/"  # Directory to store downloaded files
os.makedirs(output_dir, exist_ok=True)

keywords = ["GeneCatalog_CDS", "GeneCatalog_proteins"]       # Keywords to search for in filenames

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
    Download a file if it doesn't exist or if it exists but is 0 bytes.
    """
    file_url_full = f"https://genome.jgi.doe.gov{file_url.replace('&amp;', '&')}"
    output_file = os.path.join(output_dir, filename)

    # Check if file exists
    if os.path.exists(output_file):
        # Check if the file size is 0 bytes
        if os.path.getsize(output_file) == 0:
            print(f"File {filename} exists but is 0 bytes, re-downloading...")
        else:
            print(f"File already exists and is valid, skipping: {filename}")
            return  # File is valid, skip downloading

    # Download (either new file or re-download 0-byte file)
    print(f"Downloading: {filename}")
    download_command = [
        "curl", "-L",
        file_url_full,
        "-b", cookies_file,
        "-o", output_file
    ]
    subprocess.run(download_command, check=True)
    print(f"Saved to: {output_file}")


def download_files_from_portal(portal_name):
    """
    Fetch files.xml from a portal URL and download latest files matching keywords.
    Only downloads the latest CDS and proteins (.aa.fasta.gz) files.
    """
    #portal_name = portal_url.split("organism=")[-1]
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
        if filename.startswith(f"{portal_name}_GeneCatalog_CDS") and filename.endswith(".fasta.gz"):
            cds_files.append((filename, file_url))

        # Save only protein sequence files (.aa.fasta.gz), not annotations
        elif filename.startswith(f"{portal_name}_GeneCatalog_proteins") and filename.endswith(".aa.fasta.gz"):
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
    
    # Read the CSV file (not Excel)
    print("Reading CSV file...")
    df = pd.read_csv(excel_file)

    print(df)
    for index, row in df.iterrows():
        portal_name = str(row.get("Portal Name", "")).strip()
        if not portal_name:
            continue  # Skip empty rows
        
        # Build the portal URL directly
        #portal_url = f"https://genome.jgi.doe.gov/portal/ext-api/downloads/get-directory?organism={portal_name}"
        
        #download_files_from_portal(portal_url)
        download_files_from_portal(portal_name)
    
    print("All downloads completed!")


if __name__ == "__main__":
    main()
    
#%% resecue script
import os
import subprocess
import xml.etree.ElementTree as ET

# Configuration
output_dir = "/Volumes/Ian/FungalGenomes/"
cookies_file = "cookies"

# Reuse the same username/password from your main script
username = "ian.hsu54383@gmail.com"
password = "Yen00789"

# Helper function to log in again if needed
def login_to_jgi():
    print("Logging into JGI portal again...")
    subprocess.run([
        "curl", "https://signon.jgi.doe.gov/signon/create",
        "--data-urlencode", f"login={username}",
        "--data-urlencode", f"password={password}",
        "-c", cookies_file
    ], check=True)
    print("Login successful!")

# Find zero-byte files
def find_zero_byte_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and os.path.getsize(os.path.join(directory, f)) == 0]

# Redownload a file given its portal
def redownload_missing_file(file_name):
    # Guess portal name from file name
    # Example: "Neurospora_crassa_CDS.fasta.gz" → "Neurospora_crassa"
    portal_guess = file_name.split("_GeneCatalog")[0]

    
    print(f"Trying to recover file: {file_name} (portal guess: {portal_guess})")

    # Fetch fresh files.xml
    xml_output = os.path.join(output_dir, f"{portal_guess}_files.xml")
    fetch_command = [
        "curl", "-L",
        f"https://genome.jgi.doe.gov/portal/ext-api/downloads/get-directory?organism={portal_guess}",
        "-b", cookies_file,
        "-o", xml_output
    ]
    subprocess.run(fetch_command, check=True)

    # Parse XML
    if not os.path.exists(xml_output) or os.path.getsize(xml_output) == 0:
        print(f"Failed to fetch files.xml for {portal_guess}")
        return

    tree = ET.parse(xml_output)
    root = tree.getroot()

    for file in root.findall(".//file"):
        filename = file.attrib.get("filename", "")
        file_url = file.attrib.get("url", "")

        if filename == file_name:
            # Redownload
            file_url_full = f"https://genome.jgi.doe.gov{file_url.replace('&amp;', '&')}"
            output_path = os.path.join(output_dir, filename)
            print(f"Redownloading {filename}...")
            download_command = [
                "curl", "-L",
                file_url_full,
                "-b", cookies_file,
                "-o", output_path
            ]
            subprocess.run(download_command, check=True)
            print(f"Recovered: {filename}")
            return

    print(f"Could not find {file_name} in {portal_guess} files.xml")

def main():
    login_to_jgi()

    zero_byte_files = find_zero_byte_files(output_dir)
    print(f"Found {len(zero_byte_files)} zero-byte files.")

    for file_name in zero_byte_files:
        redownload_missing_file(file_name)

    print("Rescue complete!")

if __name__ == "__main__":
    main()

