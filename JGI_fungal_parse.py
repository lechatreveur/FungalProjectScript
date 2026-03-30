#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:08:41 2025

@author: user
"""

import pandas as pd

# Load the dataset
file_name = "/Users/user/Documents/Python_Scripts/FungalProjectScript/species_data.csv"
df = pd.read_csv(file_name)


# 1. How many species have published papers
# Assume a species has a paper if the "Published" column is not empty
species_with_paper = df['PubLink'].notna().sum()

# 2. How many unique genera (genus) are present
# Extract the first word from the "Name" column (genus)
df['Genus'] = df['Name'].apply(lambda x: x.split()[0])
unique_genera = df['Genus'].nunique()

print(f"Number of species with published papers: {species_with_paper}")
print(f"Number of unique genera: {unique_genera}")


# Keep only species with published papers
published_df = df[df['PubLink'].notna()]

# Get unique genera among published species
published_genera = published_df['Genus'].nunique()

print(f"Number of genera with at least one published paper: {published_genera}")

# Pick the first species for each genus
one_species_per_genus = published_df.drop_duplicates(subset='Genus', keep='first')

# Export to new CSV
output_file = "/Users/user/Documents/Python_Scripts/FungalProjectScript/one_species_per_genus.csv"
one_species_per_genus.to_csv(output_file, index=False)

print(f"Exported {len(one_species_per_genus)} species to {output_file}")
#%%
import pandas as pd

# Load the dataset
file_name = "/Users/user/Documents/Python_Scripts/FungalProjectScript/species_data.csv"
df = pd.read_csv(file_name)

# List of manual species
manual_species = [
    "Ustilago maydis",
    "Saccharomyces cerevisiae",
    "Pichia kudriavzevii",
    "Neurospora crassa",
    "Kluyveromyces lactis",
    "Golovinomyces cichoracearum",
    "Cryptococcus neoformans",
    "Candida albicans",
    "Blumeria graminis",
    "Aureobasidium pullulans",
    "Aspergillus nidulans",
    "Ashbya gossypii",
    "Neolecta irregularis",
    "Saitoella complicata",
    "Taphrina deformans",
    "Protomyces lactucaedebilis",
    "Schizosaccharomyces japonicus",
    "Schizosaccharomyces pombe",
    "Schizosaccharomyces cryophilus",
    "Schizosaccharomyces octosporus",
    "Lipomyces starkeyi",
    "Yarrowia lipolytica",
    "Starmerella bombicola",
    "Wickerhamiella sorbophila",
    "Tortispora caseinolytica"
]

# Filter only species with published papers
published_df = df[df['PubLink'].notna()].copy()

# Extract genus
published_df['Genus'] = published_df['Name'].apply(lambda x: x.split()[0])

# Manual species selection
manual_df = published_df[published_df['Name'].isin(manual_species)]

# Remove genera already selected by manual choices
remaining_df = published_df[~published_df['Genus'].isin(manual_df['Genus'])]

# Pick the species with the longest genome assembly per genus
remaining_one_per_genus = remaining_df.sort_values('Assembly Length', ascending=False).drop_duplicates(subset='Genus', keep='first')

# Combine manual choices with automatically selected longest assemblies
final_df = pd.concat([manual_df, remaining_one_per_genus])

# Make sure only one species per genus
final_df = final_df.drop_duplicates(subset='Genus', keep='first')

# Export
output_file = "/Users/user/Documents/Python_Scripts/FungalProjectScript/one_species_per_genus_with_manual_longest.csv"
final_df.to_csv(output_file, index=False)

print(f"Exported {len(final_df)} species to {output_file}")
