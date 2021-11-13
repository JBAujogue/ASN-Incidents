@ECHO OFF
call C:/ProgramData/Anaconda3/Scripts/activate.bat
cd %~dp0
call conda activate asn_incidents
streamlit run NuclearIncidentsUI.py --server.port 1177
call conda deactivate