# NanoMUD

NanoMUD is a deep learning based framework for detecting Uridine modifications (i.e., pseudouridine and N1-methylpseudouridine) using Nanopore direct RNA sequencing data. It offers accurate modification detection with single molecule resolution (Ψ models: min AUC = 0.86, max AUC = 0.99; m1Ψ models: min AUC = 0.87, max AUC = 0.99) and unbiased site-level stoichiometry estimation under different sequence context. NanoMUD will make a powerful tool to facilitate the research on modified therapeutic IVT RNAs and provides useful insight to the landscape and stoichiometry of pseudouridine and N1-pseudouridine on in vivo transcribed RNA species.



## Data processing

### Base calling

Make sure that your data contain the `/Analyses/Basecall` group. If not, please do local base calling with Guppy, Albacore or other base callers.

### Pod5 to Fast5

If your raw data is POD5 format, you can use the `pod5`(https://pod5-file-format.readthedocs.io/en/latest/index.html) package to convert it to fast5.

```bash
pod5 covert to_fast5 pod5/*.pod5 --output fast5/*.fast5
```

### Multi-to-single fast5

Make sure your data are in single-fast5 format, since Tombo can only take single-fast5 file as input. 

### Resquiggle

The Tombo resquiggle module will define a new assignment from raw signal to the reference sequence. Tombo documentation can be found here: https://nanoporetech.github.io/tombo/tombo.html. Please make sure you specify the `--include-event-stdev` option when running Tombo resquiggle module.

## Analysis with NanoMUD

### Feature extraction

```bash
cd /path/to/qc_final

python3 ./code/feature_extraction.py \
-i /path/to/fast5_dir \
-o /path/to/output \
-t 3 --group RawGenomeCorrected_000
```



### Read-level probability prediction

We provided fine-tuned models for psi and m1psi prediction, please select corresponding model for your interested modification type.

```bash
python3 ./code/predict_mod_probs.py \
-i ./output/tmp \ # output of feature extraction
-o ./output/probs.csv
--device cuda:0 \
--model ./model/psi/biLSTM_model \
--scaler ./model/psi/scaler
```



### Site-level stoichiometry estimation

```
python3 ./code/mod_rate_calibration.py \
-i ./output/mod_rate.csv \
--devide cuda:0 \
--model ./model/psi/regression_model
```


