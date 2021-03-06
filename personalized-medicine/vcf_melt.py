#!/usr/bin/env python
""" Melt a VCF file into a tab delimited set of calls, one per line

VCF files have all the calls from different samples on one line.  This
script reads vcf on stdin and writes all calls to stdout in tab delimited
format with one call in one sample per line.  This makes it easy to find
a given sample's genotype with, say, grep.
"""

import sys
import csv
import vcf

out = csv.writer(sys.stdout, delimiter='\t')
if len(sys.argv) > 1:
    inp = file(sys.argv[1])
else:
    inp = sys.stdin
reader = vcf.VCFReader(inp)

formats = reader.formats.keys()
infos = reader.infos.keys()

header = formats + ['FILTER', 'CHROM', 'POS', 'REF', 'ALT', 'ID'] \
        + ['info.' + x for x in infos]


out.writerow(header)


def flatten(x):
    if type(x) == type([]):
        x = ','.join(map(str, x))
    return x

for record in reader:
    info_row = [flatten(record.INFO.get(x, None)) for x in infos]
    fixed = [record.CHROM, record.POS, record.REF, record.ALT, record.ID]
    row = [record.FILTER or '.']
    row += fixed
    row += info_row
    out.writerow(row)
