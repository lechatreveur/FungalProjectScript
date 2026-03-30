#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:38:46 2025

@author: user
"""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import csv

class SimpleSpider:
    def __init__(self, base_url, max_depth=2, output_file="output.csv"):
        self.base_url = base_url
        self.visited = set()
        self.max_depth = max_depth
        self.output_file = output_file

    def normalize_url(self, url):
        """Normalize the URL by removing fragments."""
        parsed = urlparse(url)
        return urlunparse(parsed._replace(fragment=""))

    def crawl(self, url, depth=0):
        url = self.normalize_url(url)
        if depth > self.max_depth or url in self.visited:
            return

        try:
            print(f"Crawling: {url}")
            self.visited.add(url)
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            self.extract_info(soup, url)

        except requests.exceptions.RequestException as e:
            print(f"Failed to crawl {url}: {e}")
    
    def extract_info(self, soup, url):
        """Extract species details from the page and write to CSV."""
        rows = soup.find_all('tr', class_=['published', 'public'])
        data = []

        for row in rows:
            columns = row.find_all('td')
            
            # Extract details
            if len(columns) >= 4:
                species_link = columns[1].find('a')  # The link element
                name = species_link.get_text(strip=True) if species_link else "N/A"
                species_url = species_link['href'] if species_link else "N/A"
                portal_name = urlparse(species_url).path.strip('/').split('/')[-1] if species_url != "N/A" else "N/A"
                assembly_length = columns[2].get_text(strip=True)
                gene_count = columns[3].get_text(strip=True)

                # Check for publication
                publication = columns[4].find('a')
                if publication:
                    publication_text = publication.get_text(strip=True)
                    publication_link = publication['href']
                else:
                    publication_text = "No publication"
                    publication_link = "N/A"

                # Append data to the list
                data.append([name, portal_name, assembly_length, gene_count, publication_text, publication_link])

        # Write to CSV
        self.write_to_csv(data)

    def write_to_csv(self, data):
        """Write data to a CSV file."""
        with open(self.output_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)

if __name__ == "__main__":
    BASE_URL = "https://mycocosm.jgi.doe.gov/ascomycota/ascomycota.info.html"
    OUTPUT_FILE = "species_data.csv"

    # Write CSV header
    with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Portal Name", "Assembly Length", "# Genes", "Published", "PubLink"])

    # Initialize and start the spider
    spider = SimpleSpider(BASE_URL, max_depth=1, output_file=OUTPUT_FILE)
    spider.crawl(BASE_URL)


