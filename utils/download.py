import gzip
import shutil
import requests
import os
import pandas as pd


class MapDownloader:
    def __init__(self, download_path='.', verbose=False):
        """
        Initializes the downloader class.

        Parameters:
            download_path (str): Path to download files to.
            verbose (bool): If True, prints status updates.
        """
        self.download_path = download_path
        self.verbose = verbose


    def unzip_gz_file(self, gz_file_path, output_file_path):
        """
        Unzips a .gz file.

        Parameters:
            gz_file_path (str): The path to the gzipped file.
            output_file_path (str): The output path for the unzipped file.
        """
        with gzip.open(gz_file_path, 'rb') as f_in:
            with open(output_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        if self.verbose:
            print(f"Unzipped {gz_file_path} to {output_file_path}")


    def download_pdb(self, pdb_id):
        """
        Downloads a PDB file from RCSB.

        Parameters:
            pdb_id (str): The PDB ID to download.
        """
        url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
        response = requests.get(url)
        if response.status_code == 200:
            pdb_file_path = os.path.join(self.download_path, f'{pdb_id}_ref.pdb')
            with open(pdb_file_path, 'w') as file:
                file.write(response.text)
            if self.verbose:
                print(f"Downloaded PDB: {pdb_id} to {pdb_file_path}")
        else:
            print(f"Failed to download PDB: {pdb_id}")


    def download_fasta(self, pdb_id):
        """
        Downloads a FASTA file associated with a PDB ID from RCSB.

        Parameters:
            pdb_id (str): The PDB ID to download the FASTA for.
        """
        url = f'https://www.rcsb.org/fasta/entry/{pdb_id}'
        response = requests.get(url)
        if response.status_code == 200:
            fasta_file_path = os.path.join(self.download_path, f'{pdb_id}.fasta')
            with open(fasta_file_path, 'w') as file:
                file.write(response.text)
            if self.verbose:
                print(f"Downloaded FASTA for PDB ID: {pdb_id} to {fasta_file_path}")
        else:
            print(f"Failed to download FASTA for PDB ID: {pdb_id}")


    def download_emdb(self, emdb_id, delete_gz=True):
        """
        Downloads an EMDB map file and unzips it.

        Parameters:
            emdb_id (str): The EMDB ID to download.
            delete_gz (bool): If True, deletes the .gz file after extraction.
        """
        url = f'https://files.rcsb.org/pub/emdb/structures/EMD-{emdb_id}/map/emd_{emdb_id}.map.gz'
        response = requests.get(url)

        gz_file_path = os.path.join(self.download_path, f'emdb_{emdb_id}.map.gz')
        map_file_path = os.path.join(self.download_path, f'emd_{emdb_id}.map')

        if response.status_code == 200:
            with open(gz_file_path, 'wb') as file:
                file.write(response.content)
            if self.verbose:
                print(f"Downloaded EMDB: {emdb_id} to {gz_file_path}")
        else:
            print(f"Failed to download EMDB: {emdb_id}")

        # Unzip the downloaded file
        self.unzip_gz_file(gz_file_path, map_file_path)

        # Optionally delete the .gz file
        if delete_gz:
            os.remove(gz_file_path)
            if self.verbose:
                print(f"Deleted zip file: {gz_file_path}")


def main():
    mappath = '../data/raw_gan_data'
    os.makedirs(mappath, exist_ok=True)
    
    df_csv = pd.read_csv('../data/train_GAN_data.csv', dtype=str)
    columns_csv = {col: df_csv[col].tolist() for col in df_csv.columns}
    emd = columns_csv['EMID']
    pdb = columns_csv['PDBID']
    
    downloader = MapDownloader(download_path=mappath, verbose=False)
    
    for i in range(len(pdb)):    
        downloader.download_pdb(pdb[i])
        downloader.download_emdb(emd[i])
        
        print(f"Downloaded PDB-{pdb[i]} and EMDB-{emd[i]}")        
          

if __name__ == '__main__':
   main()