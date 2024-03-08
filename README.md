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
| None | 69,458,325 | 5,536,910 | 5,614,650 | 4,637,016 | 4,443,286 | 4,162,092 | 4,032,765 | 3,824,467 | 3,576,626 | 2,931,157 | 3,222,278 | 3,425,437 | 3,231,626 | 2,247,106 | 2,130,856 | 1,976,430 | 2,326,280 | 2,172,559 | 1,794,335 | 1,911,240 | 1,550,197 | 877,076 | 1,019,001 | 2,814,935 |
| 1e-0 | 69,458,325 | 5,536,910 | 5,614,650 | 4,637,016 | 4,443,286 | 4,162,092 | 4,032,765 | 3,824,467 | 3,576,626 | 2,931,157 | 3,222,278 | 3,425,437 | 3,231,626 | 2,247,106 | 2,130,856 | 1,976,430 | 2,326,280 | 2,172,559 | 1,794,335 | 1,911,240 | 1,550,197 | 877,076 | 1,019,001 | 2,814,935 |
| 1e-1 | 7,029,360 | 562,583 | 567,161 | 468,774 | 447,217 | 413,821 | 430,031 | 389,167 | 358,372 | 302,036 | 328,579 | 348,405 | 323,061 | 231,689 | 207,548 | 204,949 | 231,837 | 216,088 | 177,078 | 184,216 | 165,957 | 87,955 | 99,492 | 283,344 |
| 1e-2 | 865,268 | 69,980 | 71,951 | 57,234 | 50,108 | 49,274 | 64,353 | 48,119 | 41,130 | 36,902 | 39,393 | 42,156 | 37,330 | 27,479 | 23,996 | 26,924 | 28,890 | 29,019 | 20,499 | 21,561 | 24,746 | 10,500 | 11,741 | 31,983 |
| 1e-3 | 147,018 | 12,021 | 12,830 | 8,608 | 6,325 | 7,128 | 18,052 | 7,595 | 6,494 | 6,656 | 5,899 | 7,114 | 5,307 | 3,977 | 3,437 | 4,597 | 5,111 | 6,226 | 2,476 | 2,959 | 7,070 | 1,487 | 1,589 | 4,060 |
| 1e-4 |  44,065 |  3,232 |  4,649 | 2,384 |   877 | 1,421 |  9,575 | 1,907 | 1,832 | 2,405 | 1,115 | 2,369 | 1,253 | 498 | 821 | 1,150 | 1,763 | 659 | 286 | 613 | 3,856 | 242 | 298 | 860 |
| 1e-5 |  24,381 |  1,396 |  2,960 |   936 |    79 |   592 |  7,090 |   793 |   950 | 1,377 |   540 | 1,433 | 662 | 81 | 274 | 540 | 1,183 | 229 | 22 | 187 | 2,608 | 89 | 56 | 304 |
| 1e-6 |  17,618 |    811 |  2,468 |   593 |    22 |   286 |  5,821 |   412 |   632 | 1,047 |   362 | 1,053 | 476 | 44 | 69 | 241 | 876 | 69 | 3 | 74 | 2,075 | 57 | 0 | 127 |
| 1e-7 |  13,570 |    646 |  2,074 |   415 |    11 |   237 |  4,495 |   255 |   500 |   786 |   277 |   816 | 357 | 30 | 55 | 115 | 675 | 46 | 0 | 48 | 1,662 | 47 | 0 | 23 |
| 1e-8 |  11,133 |    570 |  1,734 |   373 |     3 |   202 |  3,866 |   216 |   447 |   668 |   196 |   495 | 297 | 23 | 47 | 96 | 541 | 7 | 0 | 31 | 1,272 | 44 | 0 | 5 |
| 1e-9 |   9,167 |    517 |  1,271 |   338 |     0 |   147 |  3,426 |   195 |   387 |   507 |   161 |   345 | 276 | 3 | 38 | 90 | 444 | 6 | 0 | 11 | 962 | 40 | 0 | 3 |
