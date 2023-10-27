@ECHO OFF
call C:\Users\jbaujogue\Anaconda3\Scripts\activate.bat
call conda activate cp_paperless
cd %~dp0..
jupyter notebook
call conda deactivate
pause