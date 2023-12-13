from __future__ import absolute_import
import os,sys,re,h5py
from statsmodels import robust
import numpy as np
import csv
import pandas as pd
import argparse
import multiprocessing
from tqdm import tqdm

def extract_read_signals(file_dir):
    read_id = file_dir.split('/')[-1].replace('.fast5', '')

    fast5_data = h5py.File(file_dir, 'r')
    group = FLAGS.group

    try:
        corr_data = fast5_data['/Analyses/'+group+'/BaseCalled_template/Events'][()]
    except:
        raise RuntimeError(('corrected data not found.'))

    aln_data = fast5_data['/Analyses/'+ group + '/BaseCalled_template/Alignment']
    aln_data = dict(list(aln_data.attrs.items()))

    fast5_data.close()
    return (read_id, aln_data, corr_data)


def extract_kmer_signals(read_id, aln_data, corr_data, kmer_filter):
    event_mean = corr_data['norm_mean']
    event_stdev = corr_data['norm_stdev']
    event_lengths = corr_data['length']
    event_bases = corr_data['base']

    seq_length = len(event_bases)
    kmers_signal = []
    for idx in range(2, seq_length - 3):

        base0, base1, base2, base3, base4 = [event_bases[idx + x].decode() for x in [-2, -1, 0, 1, 2]]
        kmer = "%s%s%s%s%s" % (base0, base1, base2, base3, base4)

        mismatches = [x.start() for x in re.finditer(kmer_filter, kmer)]
        if len(mismatches) == 0:
            continue

        mean = [event_mean[idx + x] for x in [-2, -1, 0, 1, 2]]
        std = [event_stdev[idx + x] for x in [-2, -1, 0, 1, 2]]
        length = [event_lengths[idx + x] for x in [-2, -1, 0, 1, 2]]
        chrom = [aln_data['mapped_chrom']]
        chrom_pos = [aln_data['mapped_start'] + idx]
        chrom_strand = [aln_data['mapped_strand']]
        feature = chrom + chrom_pos + chrom_strand + [read_id] + [idx] + [kmer] + mean + std + length
        kmers_signal.append(feature)

    return (kmers_signal)

def extract_file(file):
    kmer_filter = "[ACTG][ACTG]T[ACTG][ACTG]"
    try:
        read_id,aln_data,corr_data = extract_read_signals(file)
        signals = extract_kmer_signals(read_id,aln_data,corr_data,kmer_filter)
    except Exception as e:
        print(str(e))
        return None

    return signals

def iterate_files(file_list):
    if True:
        results = []
        pool = multiprocessing.Pool(processes = int(FLAGS.num_process))
        for file in file_list:
            result = pool.apply_async(extract_file, (file,))
            results.append(result)
        pool.close()

        pbar = tqdm(total=len(file_list), position=0, leave=True)
        df = pd.DataFrame()
        for result in results:
            feature = result.get()
            if feature:
                df = pd.concat([df,pd.DataFrame(feature)],axis=0)
            pbar.update(1)

        pool.join()

        return df

    else:
        df = pd.DataFrame()
        for file in file_list:
            feature = extract_file(file)
            if feature:
                df = pd.concat([df,pd.DataFrame(feature)],axis=0)
        return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract read level 5-mer features')
    parser.add_argument('-o', '--output', required=True, help='feature output dir')
    parser.add_argument('-t','--num_process', default=1, help='cpu number usage')
    parser.add_argument('-i', '--input', required=True, help='fast5 dir')
    parser.add_argument('--group', default='RawGenomeCorrected_000', help='tombo suffix')

    args = parser.parse_args(sys.argv[1:])

    global FLAGS
    FLAGS = args
    total_fl = []
    current_dir = FLAGS.input + '/'
    for current_dir, subdirs, files in os.walk(current_dir):
        for filename in files:
            if re.search(r'\.fast5', filename):
                relative_path = os.path.join(current_dir,filename)
                absolute_path = os.path.abspath(relative_path)
                total_fl.append(absolute_path.rstrip())

    os.system('mkdir -p ' + FLAGS.output + '/tmp')
    for i in range(0, len(total_fl), 5000):
        df = iterate_files(total_fl[i:i + 5000])
        df.to_csv(FLAGS.output + '/tmp/' + str(i) + '.csv', index=0)




