@ECHO OFF
call C:\Users\jb\miniconda3\Scripts\activate.bat
call conda activate asn_incidents
cd %~dp0..
streamlit run streamlit\labeling_ui.py --server.port 1453
call conda deactivate
pause