# TCGA-Onc

## Data

### Structure of Data

```bash
TACG-Onc/data/
├── bam                         # sorted aligned reads by Samtools
│   ├── SRR8924580.bam              # sorted bam file
│   ├── SRR8924580.bam.bai          # corresponding index
│   └── ...
├── bwa-mem2-2.2.1_x64-linux    # bwa-mem2, software for alignment
├── csv                         # each reads' variant num with diff p-val thresh
│   ├── SRR8924580
│   │   ├──  1.csv                      # chromosome 1
│   │   ├── ...
│   │   ├── 22.csv
│   │   └──  X.csv
│   └── ...
├── fastq                       # unaligned reads
│   ├── SRR8924580                  # sample id
│   │   ├── *R1.fastq.gz                # read 1
│   │   └── *R2.fastq.gz                # read 2
│   └── ...
├── ref
│   └── ref.fa                      # refernece genome
│   └── ...                         # corresponding index
├── sam                         # aligned reads by bwa-mem2
│   ├── SRR8924580.sam
│   └── ...
├── snp                         # genetic variation
│   ├── snp.tsv                     # raw SNPs
│   └── snp_filter.tsv              # only keep col `Chr`, `Pos`, `Pval`
└── profile.txt
```

### Unaligned Reads (FASTQ)

Raw FASTQ data address is `data/fastq/`, contain 23 samples, copy from 
`/mnt/s3/rgc-tag-onc-jh1-resources/2019_natBiotech_anneChang_bccWXS/fastq/`.
For each sample, we have two reads, i.e., `*R1.fastq.gz` and `*R2.fastq.gz`. 
The profile of these data can be find in `data/profile.txt`, copy from 
`/mnt/s3/rgc-tag-onc-jh1-resources/2019_natBiotech_anneChang_bccWXS/SraRunTable_PRJNA533341_2018_bccWXS.txt`.

To mount the S3 bucket `rgc-tag-onc-jh1-resources` (like HDD/SSD in desktop) of raw data, follow the [How to guide](https://confluence.regeneron.com/display/SUG/How+to+Install%2C+Configure%2C+and+Use+FUSE+for+S3+Mount).
First, type `id` in cmd and copy the `uid`. Then, run cmd
```bash
# usage
sudo s3fs [S3 BUCKET NAME] [MOUNT PATH] -o allow_other,use_sse=1,endpoint=us-east-1,uid=[YOUR UID],gid=1121400513,iam_role=auto

# e.g.
sudo s3fs rgc-tag-onc-jh1-resources /mnt/s3/rgc-tag-onc-jh1-resources -o allow_other,use_sse=1,endpoint=us-east-1,uid=777332657,gid=1121400513,iam_role=auto
```
Type `df -h` to check if the filesystem is mounted where `-h` means human readable. 

### Alignment (FASTQ -> SAM)

We use [bwa-mem2](https://github.com/bwa-mem2/bwa-mem2) to align the FASTQ file.
To install it,
```bash
# intall bwa-mem2
cd data
curl -L https://github.com/bwa-mem2/bwa-mem2/releases/download/v2.2.1/bwa-mem2-2.2.1_x64-linux.tar.bz2 | tar jxf -
cd ..
# index the reference genome
data/bwa-mem2-2.2.1_x64-linux/bwa-mem2 index data/ref/ref.fa
```
where the reference genome `data/ref/ref.fa` copy from 
`/mnt/efs_v2/tag_onc/users/hossein.khiabanian/ref/genome.fa`

Then, we run the alignment and store the result of each sample in 
`data/sam/*.sam`:
```bash
# usage
data/bwa-mem2-2.2.1_x64-linux/bwa-mem2 mem -t <num_threads> data/ref/ref.fa data/fastq/<ID>/*R1.fastq.gz data/fastq/<ID>/*R2.fastq.gz > data/sam/<ID>.sam

# e.g., sample `SRR8924580`, powershell
$ID = "SRR8924580"
data/bwa-mem2-2.2.1_x64-linux/bwa-mem2 mem -t 40 data/ref/ref.fa data/fastq/$ID/*R1.fastq.gz data/fastq/$ID/*R2.fastq.gz > data/sam/$ID.sam

# e.g., sample `SRR8924580` to `SRR8924602`, powershell
foreach ($i in 580..602) { 
    $ID = "SRR8924$i"; 
    data/bwa-mem2-2.2.1_x64-linux/bwa-mem2 mem -t 40 data/ref/ref.fa data/fastq/$ID/*R1.fastq.gz data/fastq/$ID/*R2.fastq.gz > data/sam/$ID.sam; 
}
```

### Convert and Sort (SAM -> BAM)

We need to convert the `.sam` file above into `.bam` file using [Samtools](https://www.htslib.org). 
To install it,
```bash
# download source file
wget https://github.com/samtools/samtools/releases/download/1.19.2/samtools-1.19.2.tar.bz2
tar -xjf samtools-1.19.2.tar.bz2
cd samtools-1.19.2
# build and install
./configure     # using default prefix /usr/local/bin/samtools
make
make install    # may need sudo for this
# check installation
samtools
which samtools  # should print /usr/local/bin/samtools
# remove source file
cd ..
rm -rf samtools-1.19.2
rm samtools-1.19.2.tar.bz2
```

Then, to convert `.sam` file into `.bam` file,
```bash
# usage
## convert SAM file to BAM file
samtools view -@ <num_threads> -S -b data/sam/<ID>.sam > data/bam/<ID>.bam
## sort the BAM file
samtools sort -@ <num_threads> data/bam/<ID>.bam -o data/bam/<ID>.bam
## index the sorted BAM file
samtools index -@ <num_threads> data/bam/<ID>.bam

# e.g., sample `SRR8924580`, powershell
$ID = "SRR8924580"
samtools view -@ 40 -S -b data/sam/$ID.sam > data/bam/$ID.bam
samtools sort -@ 40 data/bam/$ID.bam -o data/bam/$ID.bam
samtools index -@ 40 data/bam/$ID.bam

# e.g., sample `SRR8924580` to `SRR8924602`, powershell
foreach ($i in 580..602) { 
    $ID = "SRR8924$i"; 
    samtools view -@ 40 -S -b data/sam/$ID.sam > data/bam/$ID.bam;
    samtools sort -@ 40 data/bam/$ID.bam -o data/bam/$ID.bam;
    samtools index -@ 40 data/bam/$ID.bam;
}
```
We can then use [pysam](https://pysam.readthedocs.io/en/stable/#) to read the 
`.bam` file and extract the reads.

### SNPs

SNPs define genomic variant at a single base position in the DNA sequence, from
Chr 1 to 22 and X. Original SNPs file store in `data/snp/snp.tsv`, copy from 
`/mnt/s3/rgc-ag-data/app_data/yoga/Prod/gwas/817718/Meta.BCConly.COLORADO_Freeze_One__MAYO-CLINIC_Freeze_Two__SINAI_Freeze_Two__UCLA_Freeze_One__UKB_Freeze_450__UPENN-PMBB_Freeze_Two.EUR.Meta_Combined.tsv.gz`. 
It takes minutes to load whole file; so, we only keep `Chr`, `Pos`, and `Pval` 
columns and store the filtered SNPs at `data/snp/snp_filter.tsv`, which reduce
loading time to 10 seconds.
```python
import pandas as pd
snp = pd.read_csv("data/snp/snp.tsv", usecols=["Chr", "Pos", "Pval"], sep="\t")
snp.to_csv("data/snp/snp_filter.tsv", sep="\t", index=False)
```
Then, to load the filtered SNPs,
```python
import pandas as pd
snp = pd.read_csv("data/snp/snp_filter.tsv", sep="\t")
```

We can reduce the number of variant by setting a p-value threshold. Table below
shows the number of variant in total and in each chromosome with different
p-value threshold (<=),
| Pval | Total | Chr  1 | Chr  2 | Chr  3 | Chr  4 | Chr  5 | Chr  6 | Chr  7 | Chr  8 | Chr  9 | Chr 10 | Chr 11 | Chr 12 | Chr 13 | Chr 14 | Chr 15 | Chr 16 | Chr 17 | Chr 18 | Chr 19 | Chr 20 | Chr 21 | Chr 22 | Chr X |
|:----:|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| None | 65,970,269 | 5,192,193 | 5,381,111 | 4,440,612 | 4,293,765 | 3,998,634 | 3,849,370 | 3,655,832 | 3,445,147 | 2,786,893 | 3,085,913 | 3,207,132 | 3,051,380 | 2,177,228 | 2,023,149 | 1,871,722 | 2,175,604 | 1,983,958 | 1,736,268 | 1,684,923 | 1,457,013 | 835,761 | 943,503 | 2,693,158 |
| 1e-0 | 65,970,269 | 5,192,193 | 5,381,111 | 4,440,612 | 4,293,765 | 3,998,634 | 3,849,370 | 3,655,832 | 3,445,147 | 2,786,893 | 3,085,913 | 3,207,132 | 3,051,380 | 2,177,228 | 2,023,149 | 1,871,722 | 2,175,604 | 1,983,958 | 1,736,268 | 1,684,923 | 1,457,013 | 835,761 | 943,503 | 2,693,158 |
| 1e-1 | 6,781,253 | 536,961 | 550,990 | 455,233 | 437,656 | 402,913 | 416,624 | 376,996 | 349,708 | 292,190 | 319,480 | 331,557 | 310,192 | 227,428 | 200,446 | 197,171 | 220,464 | 202,259 | 173,601 | 166,596 | 158,852 | 85,156 | 94,080 | 274,700 |
| 1e-2 | 844,083 | 67,955 | 70,688 | 56,046 | 49,250 | 48,281 | 63,012 | 47,126 | 40,502 | 36,046 | 38,645 | 40,546 | 36,244 | 27,141 | 23,479 | 26,197 | 27,873 | 28,004 | 20,260 | 20,089 | 24,039 | 10,325 | 11,133 | 31,202 |
| 1e-3 | 144,451 | 11,832 | 12,663 | 8,487 | 6,224 | 7,043 | 17,848 | 7,484 | 6,433 | 6,591 | 5,805 | 6,950 | 5,170 | 3,948 | 3,369 | 4,492 | 4,927 | 6,086 | 2,448 | 2,826 | 6,913 | 1,466 | 1,505 | 3,941 |
| 1e-4 | 43,472 | 3,192 | 4,589 | 2,355 | 856 | 1,396 | 9,527 | 1,897 | 1,812 | 2,401 | 1,106 | 2,336 | 1,228 | 498 | 809 | 1,114 | 1,676 | 641 | 283 | 589 | 3,808 | 237 | 281 | 841 |
| 1e-5 | 24,188 | 1,372 | 2,932 | 936 | 79 | 588 | 7,087 | 791 | 949 | 1,374 | 536 | 1,427 | 657 | 81 | 272 | 517 | 1,128 | 223 | 22 | 170 | 2,602 | 89 | 55 | 301 |
| 1e-6 | 17,525 | 792 | 2,462 | 593 | 22 | 286 | 5,819 | 410 | 631 | 1,047 | 361 | 1,052 | 476 | 44 | 69 | 226 | 841 | 68 | 3 | 65 | 2,074 | 57 | 0 | 127 |
| 1e-7 | 13,508 | 631 | 2,072 | 415 | 11 | 237 | 4,495 | 253 | 500 | 786 | 276 | 815 | 357 | 30 | 55 | 107 | 651 | 46 | 0 | 40 | 1,661 | 47 | 0 | 23 |
| 1e-8 | 11,087 | 555 | 1,732 | 373 | 3 | 202 | 3,866 | 214 | 447 | 668 | 196 | 495 | 297 | 23 | 47 | 93 | 522 | 7 | 0 | 27 | 1,271 | 44 | 0 | 5 |
| 1e-9 | 9,128 | 503 | 1,270 | 338 | 0 | 147 | 3,426 | 193 | 387 | 507 | 161 | 345 | 276 | 3 | 38 | 89 | 425 | 6 | 0 | 10 | 961 | 40 | 0 | 3 |
