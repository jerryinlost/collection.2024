rem ---START of download.bat---
@echo off
for /F "tokens=*" %%A in  ( ids.txt) do  (
   ECHO Processing %%A.... 
   python arxiv_dl.py %%A  
)
@echo on
;
rem ---END of download.bat---