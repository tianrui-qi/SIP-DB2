# TCGA-Onc

## Todo

- check out aligned data structure and how to read them with python ([package](https://github.com/pysam-developers/pysam))
- input
    - For each sample, we have arond 100M read where average number of base per read is 124. 
    - We annote the sequencing and filter the read we select. After running [VCF](https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/DNA_Seq_Variant_Calling_Pipeline/#variant-call-annotation-workflow), select read arond place that mark as variant. - ask Nanratha
    - Then, random select same number of read from elsewhere so that we have 50% read from variant place that may directly relate to BCC (basal cell carcinoma) and 50% random read from none variant place which may same accorss all patient. `percentage: float` 
    - For the distribution of random selection, there are two way: either is select according to read of each chromosome, or we random select from all read of all chromosome. We plan to test the first one first and then use second one to increase the generality. `random: bool`
    - For each read, how to cut them according to score? - ask Nanratha `threshold: float`
- train
    - must use minibatch, still need to test out `batch_size: int`
    - fine tuning or downstring task? i.e., do we need to freeze the DNABERT2? Remember to leave a parameter to control this; to do this, write DNABERT2 in mdoel instead of dataloader so that we can track gradiate if we want. Checkout how to implement freeze gradiant. `freeze: bool`

## Data

### Structure of Data

```bash
TACG-Onc/data/
├── bam                         # convert sam to bam by Samtools
│   ├── SRR8924580.bam              # sorted bam file
│   ├── SRR8924580.bam.bai          # corresponding index
│   └── ...
├── bwa-mem2-2.2.1_x64-linux    # bwa-mem2, software for alignment
├── fastq                       # raw fastq data
│   ├── SRR8924580                  # patient id
│   │   ├── *R1.fastq.gz                # read 1
│   │   └── *R2.fastq.gz                # read 2
│   └── ...
├── ref  
│   └── ref.fa                      # refernece genome
│   └── ...                         # corresponding index
├── sam                         # alignment of fastq by bwa-mem2
│   ├── SRR8924580.sam
│   └── ...
└── profile.txt
```

### Raw FASTQ Data

Raw FASTQ data address is `data/fastq/`, contain 24 patients data, copy from 
`/mnt/s3/rgc-tag-onc-jh1-resources/2019_natBiotech_anneChang_bccWXS/fastq/`.
For each patient, we have two reads, i.e., `*R1.fastq.gz` and `*R2.fastq.gz`. 
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

### Alignment

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

Then, we run the alignment and store the result of each patient in 
`data/sam/*.sam`:
```bash
# usage
data/bwa-mem2-2.2.1_x64-linux/bwa-mem2 mem -t <num_threads> data/ref/ref.fa data/fastq/<ID>/*R1.fastq.gz data/fastq/<ID>/*R2.fastq.gz > data/sam/<ID>.sam

# e.g., patient `SRR8924580`, powershell
$ID = "SRR8924580"
data/bwa-mem2-2.2.1_x64-linux/bwa-mem2 mem -t 40 data/ref/ref.fa data/fastq/$ID/*R1.fastq.gz data/fastq/$ID/*R2.fastq.gz > data/sam/$ID.sam

# e.g., patient `SRR8924580` to `SRR8924602`, powershell
foreach ($i in 580..602) { 
    $ID = "SRR8924$i"; 
    data/bwa-mem2-2.2.1_x64-linux/bwa-mem2 mem -t 40 data/ref/ref.fa data/fastq/$ID/*R1.fastq.gz data/fastq/$ID/*R2.fastq.gz > data/sam/$ID.sam; 
}
```

### Convert SAM to BAM

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

# e.g., patient `SRR8924580`, powershell
$ID = "SRR8924580"
samtools view -@ 40 -S -b data/sam/$ID.sam > data/bam/$ID.bam
samtools sort -@ 40 data/bam/$ID.bam -o data/bam/$ID.bam
samtools index -@ 40 data/bam/$ID.bam

# e.g., patient `SRR8924580` to `SRR8924602`, powershell
foreach ($i in 580..602) { 
    $ID = "SRR8924$i"; 
    samtools view -@ 40 -S -b data/sam/$ID.sam > data/bam/$ID.bam;
    samtools sort -@ 40 data/bam/$ID.bam -o data/bam/$ID.bam;
    samtools index -@ 40 data/bam/$ID.bam;
}
```
