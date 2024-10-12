#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import argparse

def find_min_d_in_folder(folder_path, folder_c_name):
    """
    Search for files matching the naming pattern in the specified folder and extract the minimum D value.

    :param folder_path: Path to the folder (C folder)
    :param folder_c_name: Name of the folder (C folder)
    :return: The minimum D value found (if any), otherwise None
    """
    min_d = None
    # Construct the regular expression using the C folder's name
    pattern = re.compile(
        rf"^0[0-9]_target={re.escape(folder_c_name)}_distance_(-?\d+(\.\d+)?)\.png$",
        re.IGNORECASE  # Makes the pattern case-insensitive
    )

    try:
        for filename in os.listdir(folder_path):
            # Only process .png files (case-insensitive)
            if not filename.lower().endswith('.png'):
                continue

            print(f"Processing file: {filename}")  # Debugging line
            match = pattern.match(filename)
            if match:
                d_value = float(match.group(1))
                print(f"Matched! D value: {d_value}")  # Debugging line
                if (min_d is None) or (d_value < min_d):
                    min_d = d_value
            else:
                print(f"No match for file: {filename}")  # Debugging line
    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' does not exist.")
    except PermissionError:
        print(f"Error: Permission denied for folder '{folder_path}'.")
    except Exception as e:
        print(f"Unexpected error processing folder '{folder_path}': {e}")

    return min_d

def traverse_b_and_c(root_dir):
    """
    Traverse all subfolders B and their subfolders C in the root directory and record the minimum D value for each C.

    :param root_dir: Path to the root directory A
    :return: List of tuples containing (Folder B, Folder C, Min D)
    """
    results = []
    # Traverse the first level subdirectories (B)
    for b_entry in os.scandir(root_dir):
        if b_entry.is_dir():
            folder_b_name = b_entry.name
            folder_b_path = b_entry.path
            print(f"Traversing Folder B: {folder_b_name}")  # Debugging line
            # Traverse the second level subdirectories (C)
            for c_entry in os.scandir(folder_b_path):
                if c_entry.is_dir():
                    folder_c_name = c_entry.name
                    folder_c_path = c_entry.path
                    print(f"  Traversing Folder C: {folder_c_name}")  # Debugging line
                    min_d = find_min_d_in_folder(folder_c_path, folder_c_name)  # Pass C folder name
                    if min_d is not None:
                        results.append((folder_c_name, min_d))
                    else:
                        print(f"Warning: No matching files found in folder '{folder_b_name}/{folder_c_name}'.")
    return results

def write_results_to_csv(results, output_file):
    """
    Write the results to the specified output CSV file.

    :param results: List of tuples containing (Folder B, Folder C, Min D)
    :param output_file: Path to the output CSV file
    """
    # Define the header for the CSV file
    header = ['Folder B', 'Folder C', 'Min D']

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # Write the header
            for folder_c, min_d in sorted(results):
                writer.writerow([folder_c, min_d])
    except IOError as e:
        print(f"Error writing to CSV file '{output_file}': {e}")

def main():
    root_dir = "../DDMI/ffhq_brep_results"
    output_file = "../DDMI/ffhq_brep_results/ffhq_brep_knn_results.csv"

    if not os.path.isdir(root_dir):
        print(f"Error: The specified path '{root_dir}' is not a valid directory.")
        return

    results = traverse_b_and_c(root_dir)

    if results:
        write_results_to_csv(results, output_file)
    else:
        print("No matching files were found.")

if __name__ == "__main__":
    main()
