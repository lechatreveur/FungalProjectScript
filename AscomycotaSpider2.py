#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:35:27 2025

@author: user
"""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

class SpeciesTextSpider:
    def __init__(self, base_url, output_dir="/Users/user/Documents/FungalProject/species_text", max_depth=1):
        self.base_url = base_url
        self.visited = set()
        self.output_dir = output_dir
        self.max_depth = max_depth

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def normalize_url(self, url):
        """Normalize the URL by removing fragments."""
        parsed = urlparse(url)
        return parsed.scheme + "://" + parsed.netloc + parsed.path

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

            self.extract_species_links(soup)
        except requests.exceptions.RequestException as e:
            print(f"Failed to crawl {url}: {e}")

    def extract_species_links(self, soup):
        """Extract links to species pages."""
        rows = soup.find_all('tr', class_=['published', 'public'])
        for row in rows:
            species_link = row.find('a')
            if species_link:
                species_url = urljoin(self.base_url, species_link['href'])
                portal_name = urlparse(species_url).path.strip('/').split('/')[-1]
                species_page = f"{species_url}/{portal_name}.home.html"
                self.scrape_species_page(species_page, portal_name)

    def scrape_species_page(self, url, portal_name):
        """Scrape text data from a species' page."""
        try:
            print(f"Scraping species page: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract text
            description_div = soup.find('div', class_='home')
            description_text = description_div.get_text(strip=True) if description_div else "No description available."

            # Save text data
            self.save_text(portal_name, description_text)

        except requests.exceptions.RequestException as e:
            print(f"Failed to scrape species page {url}: {e}")

    def save_text(self, portal_name, description_text):
        """Save the extracted text."""
        text_file_path = os.path.join(self.output_dir, f"{portal_name}.txt")
        with open(text_file_path, 'w', encoding='utf-8') as file:
            file.write(description_text)
        print(f"Text saved: {text_file_path}")

if __name__ == "__main__":
    BASE_URL = "https://mycocosm.jgi.doe.gov/ascomycota/ascomycota.info.html"
    spider = SpeciesTextSpider(BASE_URL, output_dir="species_text", max_depth=1)
    spider.crawl(BASE_URL)


#%% download images and text
# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin, urlparse
# import csv
# import os

# class SpeciesSpider:
#     def __init__(self, base_url, output_dir="/RAID1/working/R402/hsushen/FungalProject/JGIspecies_data", max_depth=1):
#         self.base_url = base_url
#         self.visited = set()
#         self.output_dir = output_dir
#         self.max_depth = max_depth

#         # Create output directory if it doesn't exist
#         os.makedirs(self.output_dir, exist_ok=True)

#     def normalize_url(self, url):
#         """Normalize the URL by removing fragments."""
#         parsed = urlparse(url)
#         return parsed.scheme + "://" + parsed.netloc + parsed.path

#     def crawl(self, url, depth=0):
#         url = self.normalize_url(url)
#         if depth > self.max_depth or url in self.visited:
#             return

#         try:
#             print(f"Crawling: {url}")
#             self.visited.add(url)
#             response = requests.get(url, timeout=10)
#             response.raise_for_status()
#             soup = BeautifulSoup(response.text, 'html.parser')

#             self.extract_species_links(soup)
#         except requests.exceptions.RequestException as e:
#             print(f"Failed to crawl {url}: {e}")

#     def extract_species_links(self, soup):
#         """Extract links to species pages."""
#         rows = soup.find_all('tr', class_=['published', 'public'])
#         for row in rows:
#             species_link = row.find('a')
#             if species_link:
#                 species_url = urljoin(self.base_url, species_link['href'])
#                 portal_name = urlparse(species_url).path.strip('/').split('/')[-1]
#                 species_page = f"{species_url}/{portal_name}.home.html"
#                 self.scrape_species_page(species_page, portal_name)

#     def scrape_species_page(self, url, portal_name):
#         """Scrape data from a species' page."""
#         try:
#             print(f"Scraping species page: {url}")
#             response = requests.get(url, timeout=10)
#             response.raise_for_status()
#             soup = BeautifulSoup(response.text, 'html.parser')

#             # Extract text
#             description_div = soup.find('div', class_='home')
#             description_text = description_div.get_text(strip=True) if description_div else "No description available."

#             # Extract image
#             image_tag = soup.find('div', class_='picture').find('img') if soup.find('div', class_='picture') else None
#             if image_tag:
#                 image_url = urljoin(self.base_url, image_tag['src'])
#                 image_caption = image_tag.get('alt', 'No caption')
#                 self.download_image(image_url, portal_name)

#             # Save data
#             self.save_text(portal_name, description_text, image_caption if image_tag else None)

#         except requests.exceptions.RequestException as e:
#             print(f"Failed to scrape species page {url}: {e}")

#     def download_image(self, image_url, portal_name):
#         """Download an image and save it."""
#         try:
#             print(f"Downloading image: {image_url}")
#             response = requests.get(image_url, timeout=10)
#             response.raise_for_status()
#             image_path = os.path.join(self.output_dir, f"{portal_name}.jpg")
#             with open(image_path, 'wb') as file:
#                 file.write(response.content)
#             print(f"Image saved: {image_path}")
#         except requests.exceptions.RequestException as e:
#             print(f"Failed to download image {image_url}: {e}")

#     def save_text(self, portal_name, description_text, image_caption=None):
#         """Save the extracted text and image caption."""
#         text_file_path = os.path.join(self.output_dir, f"{portal_name}.txt")
#         with open(text_file_path, 'w', encoding='utf-8') as file:
#             file.write(f"Description:\n{description_text}\n")
#             if image_caption:
#                 file.write(f"\nImage Caption:\n{image_caption}\n")
#         print(f"Text saved: {text_file_path}")

# if __name__ == "__main__":
#     BASE_URL = "https://mycocosm.jgi.doe.gov/ascomycota/ascomycota.info.html"
#     spider = SpeciesSpider(BASE_URL, output_dir="species_data", max_depth=1)
#     spider.crawl(BASE_URL)
