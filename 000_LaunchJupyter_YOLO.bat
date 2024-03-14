call C:\Users\%USERNAME%\anaconda3\Scripts\activate.bat

call conda activate yolo

cd %~dp0

start cmd.exe /k "cd %~dp0"

call jupyter notebook

cmd /k