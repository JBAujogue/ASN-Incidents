@ECHO OFF
call C:\Users\jb\miniconda3\Scripts\activate.bat
call conda activate asn_incidents
cd %~dp0..
streamlit run streamlit\nuclear_incidents_UI.py --server.port 1177
call conda deactivate
pause