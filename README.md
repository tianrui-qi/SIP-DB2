# TCGA-Onc

## Todo

- aligning raw data with bwa
- check out aligned data structure and how to read them with python

## Data

### Raw data

Raw data address: `/mnt/s3/rgc-tag-onc-jh1-resources/2019_natBiotech_anneChang_bccWXS/fastq/`

To mount the S3 bucket (like HDD/SSD in desktop) of raw data, follow the [How to guide](https://confluence.regeneron.com/display/SUG/How+to+Install%2C+Configure%2C+and+Use+FUSE+for+S3+Mount).
First, type `id` in cmd and copy the `uid`. Run cmd
```bash
sudo s3fs [S3 BUCKET NAME] [MOUNT PATH] -o allow_other,use_sse=1,endpoint=us-east-1,uid=[YOUR UID],gid=1121400513,iam_role=auto
```
i.e.
```bash
sudo s3fs rgc-tag-onc-jh1-resources /mnt/s3/rgc-tag-onc-jh1-resources -o allow_other,use_sse=1,endpoint=us-east-1,uid=777332657,gid=1121400513,iam_role=auto
```
Then, type `df -h` to check if the filesystem is mounted where `-h` means human readable. 

### Alignment


