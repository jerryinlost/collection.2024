Assume that you collected some files containing arxiv urls in a directory named 'references'

```sh
# Extract ids
# use shell commands
grep -nr 'arxiv.org' references | grep -Eo 'https://arxiv.org/(abs|pdf|abs/pdf)/[0-9]+\.[0-9]+' >> ids.txt
# or
# use shell and python commands
grep -nr 'arxiv.org' >> all.txt 
python extract.py >> ids.txt
rm all.txt
```

```sh
# Remove duplicates
sort "ids.txt" | uniq -u >> unique_ids.txt
```

```sh
# Download arxiv papers
cat "unique_ids.txt" | while read line; do python arxiv_dl.py $line; done

# use this command when you experienced network breakdown.
cat "leftover.lst" | while read line; do python arxiv_dl.py $line; done
```

```sh
# get done list
ls  |grep pdf|grep -Eo '[0-9]+\.[0-9]+' |linecount
ls  |grep pdf|grep -Eo '[0-9]+\.[0-9]+' >> done.lst

# get the leftovers
grep -vxFf done.lst unique_ids.lst |linecount
grep -vxFf done.lst unique_ids.lst > leftover.lst
```