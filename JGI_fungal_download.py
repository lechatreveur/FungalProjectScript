#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:38:55 2024

@author: user
"""


#%% download 
import pandas as pd
import json
import subprocess
import os


# Fonction pour obtenir l'ID de l'organisme
def get_organism_id(genome, csrf_token):
    search_query = f"https://files.jgi.doe.gov/search/?q={genome.replace(' ', '%20')}&a=false&h=false&d=asc&p=1&x=10&t=simple&ff%5Bfile_type%5D=fasta&api_version=2"
    output_file = "output.json"

    # Commande pour obtenir les résultats de la recherche
    curl_search_command = [
        "curl", "-X", "GET", search_query,
        "-H", "accept: application/json",
        "-H", f"X-CSRFToken: {csrf_token}",
        "-o", output_file
    ]
    
    # Exécution de la commande curl
    subprocess.run(curl_search_command, capture_output=True, text=True)

    # Commande pour extraire l'ID de l'organisme utilisant jq
    jq_organism_id_command = [
        "jq", "-r", '.organisms[0].id', output_file
    ]
    
    # Exécution de la commande jq pour obtenir l'ID de l'organisme
    result = subprocess.run(jq_organism_id_command, capture_output=True, text=True)
    organism_id = result.stdout.strip()

    return organism_id

# # Fonction pour obtenir les IDs des fichiers
# def get_file_ids(genome, csrf_token):
#     search_query = f"https://files.jgi.doe.gov/search/?q={genome.replace(' ', '%20')}&a=false&h=false&d=asc&p=1&x=10&t=simple&ff%5Bfile_type%5D=fasta&api_version=2"
#     output_file = "output.json"

#     # Commande pour obtenir les résultats de la recherche
#     curl_search_command = [
#         "curl", "-X", "GET", search_query,
#         "-H", "accept: application/json",
#         "-H", f"X-CSRFToken: {csrf_token}",
#         "-o", output_file
#     ]
    
#     # Exécution de la commande curl
#     subprocess.run(curl_search_command, capture_output=True, text=True)

#     # Commande pour extraire les IDs des fichiers utilisant jq
#     jq_extract_command = [
#         "jq", "-r", '.organisms[] | .files[] | select(.file_name | contains("protein") or contains("cds")) | ._id', output_file
#     ]
    
#     # Exécution de la commande jq pour extraire les IDs de fichier
#     result = subprocess.run(jq_extract_command, capture_output=True, text=True)
#     file_ids = result.stdout.strip().split('\n')
#     file_ids = [id for id in file_ids if id]

#     return file_ids
def get_file_ids(genome, csrf_token):
    search_query = f"https://files.jgi.doe.gov/search/?q={genome.replace(' ', '%20')}&api_version=2"
    output_file = "output.json"

    # Run the curl command
    curl_search_command = [
        "curl", "-X", "GET", search_query,
        "-H", "accept: application/json",
        "-H", f"X-CSRFToken: {csrf_token}",
        "-o", output_file
    ]
    subprocess.run(curl_search_command, capture_output=True, text=True)

    # Debugging: Inspect the output JSON
    with open(output_file, 'r') as f:
        print(f"Debugging {genome}: {json.load(f)}")

    # Adjusted jq filter for fungal genomes
    jq_extract_command = [
        "jq", "-r", '.organisms[] | .files[] | select(.file_name | contains("genome") or contains("annotation")) | ._id', output_file
    ]
    result = subprocess.run(jq_extract_command, capture_output=True, text=True)
    file_ids = result.stdout.strip().split('\n')
    return [id for id in file_ids if id]

# Fonction pour écrire le payload JSON
def ecrire_payload(genome,organisms_ids, file_ids, save_dir, nom_fichier="payload.json"):
    """
    Génère un payload JSON pour la requête API de téléchargement de fichiers 
    et l'écrit dans un fichier.

    Args:
    - organisms_ids (str): ID de l'organisme.
    - file_ids (list): Liste d'IDs de fichiers associés à cet organisme.
    - nom_fichier (str): Nom du fichier où le payload JSON sera sauvegardé.

    Returns:
    - None
    """

    # Créer le dictionnaire pour le payload JSON
    ids_dict = {
        organisms_ids: file_ids
    }

    # Ajouter la version de l'API
    payload = {
        "ids": ids_dict,
        "api_version": "2"
    }
    #print(payload)
    # Convertir le payload en JSON
    payload_json = json.dumps(payload, indent=4)
    
    # Écrire le payload dans un fichier
    with open(nom_fichier, 'w') as fichier:
        fichier.write(payload_json)
    
    print(f"Payload JSON écrit dans {nom_fichier} avec succès.")
    # Commande curl pour télécharger les fichiers
    curl_command = [
    "curl",
    "-X", "POST", "https://files-download.jgi.doe.gov/download_files/",
    "-H", "accept: application/json",
    "-H", f"Authorization: {auth_token}",  # Remplacez {csrf_token} par le token réel
    "-H", "Content-Type: application/json",
    "-d", payload_json,#"{payload}",#f"@{nom_fichier}"
    "--output", f"/Users/user/Documents/PlantProject/genome/{genome}.zip"  # Nom du fichier de sortie
    ]
    

    # Exécuter la commande curl
    subprocess.run(curl_command)

# Exemple de DataFrame avec des génomes (pour l'exemple, df est supposé être défini)
# df = pd.DataFrame({'genome': ['genome1', 'genome2']})

# Définir le répertoire de sauvegarde pour les génomes
#save_dir = '/Users/user/Documents/PlantProject/genome/'
save_dir = '/Users/user/Documents/FungalProject/genome/'
os.makedirs(save_dir, exist_ok=True)
# Charger le fichier Excel
#file_path = '/Users/user/Documents/PlantProject/PlantGenomesToPick.xlsx'
file_path = '/Users/user/Documents/FungalProject/FungalGenomesToPick.xlsx'
df = pd.read_excel(file_path)

# Filtrer les lignes où le genome est non null
df = df.dropna(subset=['genome'])

# Votre token d'authentification
auth_token = "Bearer /api/sessions/53eb4c545724ecda350450533e1b87a3"  # Remplacez par votre token réel
csrf_token = "OgYq0qqkdOqJnwJuZAfjwCNn2vjJJ8nISdzExgX4MB4fOnqDUsQR3j3DJpYiY6ey"  # Remplacez par votre token CSRF réel




# Itérer sur chaque génome et traiter les requêtes
for genome in df['genome']:
    print(f"Processing genome: {genome}")
    file_ids = get_file_ids(genome, csrf_token)
    organism_id = get_organism_id(genome, csrf_token)
    if file_ids and organism_id:
        ecrire_payload(genome,organism_id, file_ids, save_dir)#, nom_fichier=f"payload_{genome}.json")
    else:
        print(f"No relevant files found for genome: {genome}")

# curl --cookie jgi_session=/api/sessions/21039a42f826d0ae12715c4f173ae0bb --output download.20240809.115620.zip -d "{\"ids\":{\"Phytozome-277\":[\"53112afb49607a1be0055a4d\",\"53112af949607a1be0055a49\",\"53112af849607a1be0055a46\",\"53112afb49607a1be0055a4e\"]}}" -H "Content-Type: application/json" https://files-download.jgi.doe.gov/filedownload/
# curl --cookie jgi_session=/api/sessions/21039a42f826d0ae12715c4f173ae0bb --output download.20240809.115620.zip -d "{\"ids\":{\"Phytozome-277\":[\"53112afb49607a1be0055a4d\",\"53112af949607a1be0055a49\",\"53112af849607a1be0055a46\",\"53112afb49607a1be0055a4e\"]}}" -H "Content-Type: application/json" https://files-download.jgi.doe.gov/filedownload/
# curl -X POST "https://files-download.jgi.doe.gov/download_files/" -H "accept: application/json" -H "Authorization: OgYq0qqkdOqJnwJuZAfjwCNn2vjJJ8nISdzExgX4MB4fOnqDUsQR3j3DJpYiY6ey" -H "Content-Type: application/json" -d "{\"ids\": {\"Phytozome-296\":[ \"54ad8ddb0d8785565d4707ae\", \"56901b8e0d878508e3d1fb88\", \"56901b920d878508e3d1fb8a\", \"54ad8ddb0d8785565d4707ad\", \"54ad8ddb0d8785565d4707b1\", \"54ad8ddb0d8785565d4707b2\", \"56901b950d878508e3d1fb8c\", \"56901b950d878508e3d1fb8b\"]}, \"api_version\": \"2\"}" --output /Users/user/Documents/PlantProject/genome/Triticum_aestivum_v2.2_downloaded_files.zip
#curl -X POST "https://files-download.jgi.doe.gov/download_files/" -H "accept: application/json" -H "Authorization: /api/sessions/ad6b46e4270c40746c9bae319860c9d7" -H "Content-Type: application/json" -d "{\"ids\": {\"Phytozome-296\":[ \"54ad8ddb0d8785565d4707ae\", \"56901b8e0d878508e3d1fb88\", \"56901b920d878508e3d1fb8a\", \"54ad8ddb0d8785565d4707ad\", \"54ad8ddb0d8785565d4707b1\", \"54ad8ddb0d8785565d4707b2\", \"56901b950d878508e3d1fb8c\", \"56901b950d878508e3d1fb8b\"]}, \"api_version\": \"2\"}" --output /Users/user/Documents/PlantProject/genome/Triticum_aestivum_v2.2_downloaded_files.zip
#'curl', '-X', 'POST', 'https://files-download.jgi.doe.gov/download_files/', '-H', 'accept: application/json', '-H', 'Authorization: /api/sessions/21039a42f826d0ae12715c4f173ae0bb', '-H', 'Content-Type: application/json', '-d', "{'ids': {'Phytozome-322': ['569335620d87851ee9726ac1', '569335620d87851ee9726ac6', '569335600d87851ee9726ac0', '569335620d87851ee9726ac7']}, 'api_version': '2'}", '--output', '/Users/user/Documents/PlantProject/genome/Aquilegia coerulea v3.1_downloaded_files.zip'
#curl -X POST "https://files-download.jgi.doe.gov/filedownload/" -H "accept: application/json" -H "Authorization: OgYq0qqkdOqJnwJuZAfjwCNn2vjJJ8nISdzExgX4MB4fOnqDUsQR3j3DJpYiY6ey" -H "Content-Type: application/json" -d "{\"ids\": {\"Phytozome-296\":[ \"54ad8ddb0d8785565d4707ae\", \"56901b8e0d878508e3d1fb88\", \"56901b920d878508e3d1fb8a\", \"54ad8ddb0d8785565d4707ad\", \"54ad8ddb0d8785565d4707b1\", \"54ad8ddb0d8785565d4707b2\", \"56901b950d878508e3d1fb8c\", \"56901b950d878508e3d1fb8b\"]}, \"api_version\": \"2\"}" --output /Users/user/Documents/PlantProject/genome/Triticum_aestivum_v2.2_downloaded_files.zip
#curl --cookie jgi_session=/api/sessions/21039a42f826d0ae12715c4f173ae0bb --output download.20240809.115620.zip -d "{\"ids\":{\"Phytozome-277\":[\"53112afb49607a1be0055a4d\",\"53112af949607a1be0055a49\",\"53112af849607a1be0055a46\",\"53112afb49607a1be0055a4e\"]}}" -H "Content-Type: application/json" https://files-download.jgi.doe.gov/filedownload/
#curl -X POST "https://files-download.jgi.doe.gov/download_files/" -H "accept: application/json" -H "Authorization: Bearer /api/sessions/ad6b46e4270c40746c9bae319860c9d7" -H "Content-Type: application/json" -d "{\"ids\": {\"Phytozome-296\":[ \"54ad8ddb0d8785565d4707ae\", \"56901b8e0d878508e3d1fb88\", \"56901b920d878508e3d1fb8a\", \"54ad8ddb0d8785565d4707ad\", \"54ad8ddb0d8785565d4707b1\", \"54ad8ddb0d8785565d4707b2\", \"56901b950d878508e3d1fb8c\", \"56901b950d878508e3d1fb8b\"]}, \"api_version\": \"2\"}" --output /Users/user/Documents/PlantProject/genome/Triticum_aestivum_v2.2_downloaded_files.zip