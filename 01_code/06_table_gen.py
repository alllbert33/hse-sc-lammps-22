#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Microarchitecture Summary Table
===============================

This script loads a benchmark CSV file, filters by socket count (1, 2, or 4),
maps detailed microarchitecture names to broader families (Intel and AMD only),
and generates a pivot table showing:
- Number of unique processor configurations (Count)
- Total number of public test runs (TotalRuns)
for each microarchitecture family, broken down by 1S, 2S, 4S.

The results are printed to console and saved to a CSV file in ../03_data/tab/
"""

import pandas as pd
import re
import numpy as np
import os

# ===================== 0. PATHS =====================
# Determine script directory and data directories
script_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.normpath(os.path.join(script_dir, '..', '03_data'))
output_dir = os.path.join(data_root, 'tab')
os.makedirs(output_dir, exist_ok=True)

# Input file (assumed to be directly in 03_data)
input_file = os.path.join(data_root, 'openbenchmarking - export.csv')
# If you have a different file name, change it here, e.g. 'openbenchmarking - Лист29.csv'

# ---------------------------------------------------------------------
# 1. LOAD AND PREPARE
# ---------------------------------------------------------------------
df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()

required = ['Microarchitecture', 'Socket(s)', '# Compatible Public Results']
for col in required:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found")

# ---------------------------------------------------------------------
# 2. PROCESS SOCKET(S)
# ---------------------------------------------------------------------
df['Socket(s)'] = pd.to_numeric(df['Socket(s)'], errors='coerce')
df = df.dropna(subset=['Socket(s)'])
df = df[df['Socket(s)'].isin([1, 2, 4])].copy()
df['Socket(s)'] = df['Socket(s)'].astype(int)

# ---------------------------------------------------------------------
# 3. MAP MICROARCHITECTURES TO FAMILIES
# ---------------------------------------------------------------------
def map_microarch(name):
    """Map detailed microarchitecture names to broader families."""
    if pd.isna(name):
        return 'Unknown'
    s = str(name).strip()
    slow = s.lower()

    # Intel – Skylake / Kaby Lake
    if ('skylake' in slow.replace(' ', '') or 'sky lake' in slow or
        'kabylake' in slow.replace(' ', '') or 'kaby lake' in slow):
        return 'Intel Skylake / Kaby Lake'
    if 'cascade' in slow and 'lake' in slow:
        return 'Intel Cascade Lake'
    if 'comet' in slow and 'lake' in slow:
        return 'Intel Comet Lake'
    if 'ice' in slow and 'lake' in slow:
        return 'Intel Ice Lake'
    if ('alder' in slow or 'raptor' in slow) and 'lake' in slow:
        return 'Intel Alder Lake / Raptor Lake'
    if 'sapphire' in slow and 'rapids' in slow:
        return 'Intel Sapphire Rapids'
    if 'emerald' in slow and 'rapids' in slow:
        return 'Intel Emerald Rapids'
    if 'granite' in slow and 'rapids' in slow:
        return 'Intel Granite Rapids'
    if ('meteor' in slow or 'arrow' in slow) and 'lake' in slow:
        return 'Intel Meteor Lake / Arrow Lake'

    # AMD
    if 'zen' in slow:
        if re.search(r'zen\s*2', slow):
            return 'AMD Zen 2'
        if re.search(r'zen\s*3', slow):
            return 'AMD Zen 3'
        if re.search(r'zen\s*4', slow):
            return 'AMD Zen 4'
        if re.search(r'zen\s*5', slow):
            return 'AMD Zen 5'
        return 'AMD Zen / Zen+'

    # Everything else marked as Unknown (will be filtered out)
    return 'Unknown'

df['MicroarchGroup'] = df['Microarchitecture'].apply(map_microarch)

# ---------------------------------------------------------------------
# 4. KEEP ONLY INTEL AND AMD (exclude Neoverse-V2, Unknown, etc.)
# ---------------------------------------------------------------------
intel_amd_groups = [
    'Intel Skylake / Kaby Lake',
    'Intel Cascade Lake',
    'Intel Comet Lake',
    'Intel Ice Lake',
    'Intel Alder Lake / Raptor Lake',
    'Intel Sapphire Rapids',
    'Intel Emerald Rapids',
    'Intel Granite Rapids',
    'Intel Meteor Lake / Arrow Lake',
    'AMD Zen / Zen+',
    'AMD Zen 2',
    'AMD Zen 3',
    'AMD Zen 4',
    'AMD Zen 5'
]

df = df[df['MicroarchGroup'].isin(intel_amd_groups)]

# ---------------------------------------------------------------------
# 5. AGGREGATION
# ---------------------------------------------------------------------
grouped = df.groupby(['MicroarchGroup', 'Socket(s)']).agg(
    Count=('Microarchitecture', 'size'),
    TotalRuns=('# Compatible Public Results', 'sum')
).reset_index()

# ---------------------------------------------------------------------
# 6. PIVOT TABLE
# ---------------------------------------------------------------------
pivot = grouped.pivot(index='MicroarchGroup', columns='Socket(s)',
                      values=['Count', 'TotalRuns']).fillna(0)

# Rename columns
new_columns = []
for metric, socket_val in pivot.columns:
    socket_int = int(socket_val)
    if metric == 'Count':
        new_columns.append(f'{socket_int}S')
    elif metric == 'TotalRuns':
        new_columns.append(f'{socket_int}Snum')
pivot.columns = new_columns

# Add missing columns if some socket counts are absent
for col in ['1S', '2S', '4S', '1Snum', '2Snum', '4Snum']:
    if col not in pivot.columns:
        pivot[col] = 0

# Set desired column order
pivot = pivot[['1S', '2S', '4S', '1Snum', '2Snum', '4Snum']]
pivot['Total'] = pivot['1S'] + pivot['2S'] + pivot['4S']
pivot['Totalnum'] = pivot['1Snum'] + pivot['2Snum'] + pivot['4Snum']
pivot = pivot[['1S', '2S', '4S', 'Total', '1Snum', '2Snum', '4Snum', 'Totalnum']]

pivot = pivot.astype(int)

# ---------------------------------------------------------------------
# 7. ROW ORDER AND TOTAL ROW
# ---------------------------------------------------------------------
desired_order = intel_amd_groups
extra = sorted(set(pivot.index) - set(desired_order))
final_index = [g for g in desired_order if g in pivot.index] + extra
pivot = pivot.reindex(final_index, fill_value=0)

total_row = pivot.sum().astype(int)
total_row.name = 'Total'
pivot = pd.concat([pivot, total_row.to_frame().T])

# ---------------------------------------------------------------------
# 8. OUTPUT
# ---------------------------------------------------------------------
print(pivot.to_string())

# Save to CSV in the output directory
output_csv = os.path.join(output_dir, 'microarchitecture_summary.csv')
pivot.to_csv(output_csv)
print(f"\nTable saved to: {os.path.basename(output_csv)} (in folder: {os.path.basename(output_dir)})")