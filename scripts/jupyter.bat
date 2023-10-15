@ECHO OFF
call C:\Users\jb\miniconda3\Scripts\activate.bat
call conda activate asn_incidents
cd %~dp0..
jupyter notebook
call conda deactivate
pause