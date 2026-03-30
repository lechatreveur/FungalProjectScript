#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:34:38 2024

@author: user
"""

import os
import requests
import sys

def format_species_name(species_name):
    """Correctly format the species name to replace spaces with underscores."""
    return species_name.replace(" ", "_")

def fetch_genomes_by_taxonomy(taxonomy_name):
    """Fetch genome data for all species under a given taxonomy from Ensembl Fungi."""
    url = f"http://rest.ensembl.org/info/genomes/taxonomy/{taxonomy_name}?content-type=application/json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data for taxonomy: {taxonomy_name}")
        return []

def get_internal_name(species_info, target_species):
    """Find internal name for a specific species using wildcard matching."""
    target_species_lower = target_species.lower().replace(' ', '_')
    for entry in species_info:
        if target_species_lower in entry['name']:
            return entry['name']
    return "Internal name not found"

def fetch_genome_info(species_name):
    """ Fetch genome information from Ensembl REST API. """
    url = f"http://rest.ensembl.org/info/assembly/{species_name}?content-type=application/json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch genome information for {species_name}: {response.text}")

def download_chromosome_sequence(species_name, chromosome_name, start, end):
    """ Download sequence data for a specified region using start and end points. """
    region = f"{chromosome_name}:{start}..{end}"
    url = f"http://rest.ensembl.org/sequence/region/{species_name}/{region}?content-type=text/plain"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to download sequence for {chromosome_name}: {response.text}")

def download_genome(species_name):
    """Download the entire genome for the specified species by parts, handling each chromosome separately."""
    species_name_corrected = format_species_name(species_name)
    genome_info = fetch_genome_info(species_name_corrected)
    sequences = {}

    total_steps = sum((region['length'] // 10000000 + (1 if region['length'] % 10000000 != 0 else 0)
                       for region in genome_info['top_level_region']))
    current_step = 0

    for region in genome_info['top_level_region']:
        chromosome_name = region['name']
        chromosome_length = region['length']
        step_size = 10000000  # 10Mb

        for start in range(1, chromosome_length, step_size):
            end = min(start + step_size - 1, chromosome_length)
            sequence = download_chromosome_sequence(species_name_corrected, chromosome_name, start, end)
            if chromosome_name in sequences:
                sequences[chromosome_name] += sequence
            else:
                sequences[chromosome_name] = sequence
            
            current_step += 1
            sys.stdout.write(f'\rDownloading {species_name}: {current_step}/{total_steps} parts completed.')
            sys.stdout.flush()

    return sequences

def download_annotations(species_name):
    """Download the annotation file (GFF) for the specified species."""
    species_name_corrected = format_species_name(species_name)
    url = f"http://rest.ensembl.org/overlap/region/{species_name_corrected}?feature=gene;content-type=application/gff3"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to download annotations for {species_name}: {response.text}")

def main():
    species_info = fetch_genomes_by_taxonomy("ascomycota")
    with open("/home/hsushen/FungalProjectScript/species_list.txt", "r") as file:
        species_list = [line.strip() for line in file]

    for species in species_list:
        internal_name = get_internal_name(species_info, species)
        if "Internal name not found" not in internal_name:
            print(f"Processing genome for {internal_name}...")
            try:
                genome_sequences = download_genome(internal_name)
                db_dir = f"/RAID1/working/R402/hsushen/FungalProject/MolEvo/Database/{internal_name}/"
                os.makedirs(db_dir, exist_ok=True)
                for chromo, seq in genome_sequences.items():
                    filename = os.path.join(internal_name, f"{chromo}.fasta")
                    with open(filename, 'w') as file:
                        file.write(f">{chromo}\n{seq}\n")
                    print(f"Saved {chromo} to {filename}")
                
                annotations = download_annotations(internal_name)
                annotation_file = os.path.join(db_dir, f"{internal_name}_annotations.gff")
                with open(annotation_file, 'w') as file:
                    for feature in annotations:
                        file.write(f"{feature}\n")
                    print(f"Saved annotations to {annotation_file}")
            except Exception as e:
                print(e)
        else:
            print(f"{species}: {internal_name}")

if __name__ == "__main__":
    main()
