#!/usr/bin/env python
"""
Takes a tsv file (CADD output) and a vcf from 1kg project
and filters out the 1kg variants from our identified variants.
"""

import vcf
import sys

filtered_cadd = sys.argv[1]
clinvar_f = sys.argv[2]
outfile = sys.argv[3]

clinvar = vcf.Reader(open(clinvar_f, 'r'))

clinvar_vars = {} 

# gather all 1000g indels
i = 0
for record in clinvar:
    i += 1
    chr_pos = (str(record.CHROM), int(record.POS))
    clinvar_vars[chr_pos] = (record.REF, record.ALT[0]) # record.ALT is a list
    if i % 10000 == 0:
        print i

print
print len(clinvar_vars)
print

writer = open(outfile, 'w')
removed = 0
# check if there are any identical mutations in the same position
with open(filtered_cadd, 'r') as cadd:
    next(cadd)
    next(cadd)
    for line in cadd:
        spl = line.split()
        chr_pos = (str(spl[0]), int(spl[1]))
        if chr_pos in clinvar_vars:
            print chr_pos
            if clinvar_vars[chr_pos][0] == spl[2] and clinvar_vars[chr_pos][1]  == spl[3]:
                removed += 1
                writer.write(line)
writer.close()

print "Our mutations identified in CLINVAR: " + str(removed)

