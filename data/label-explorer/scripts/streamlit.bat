@ECHO OFF
call C:\Users\jbaujogue\Anaconda3\Scripts\activate.bat
call conda activate cp_paperless
cd %~dp0..
streamlit run streamlit\explorer.py --server.port 1177
call conda deactivate
pause