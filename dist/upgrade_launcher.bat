IF NOT EXIST "doclayout\doclayout.exe" (
    tar -xf doclayout.zip
)

gitbigfileftp.exe /DownloadRemote /NoTipInfo /NoCheckFile .
IF %ERRORLEVEL% EQU 0 (
    echo 程序成功执行，返回值为 0
) ELSE (
    echo 程序执行出现问题，返回值为 %ERRORLEVEL%
    timeout /T 5 /NOBREAK >nul
    gitbigfileftp.exe /DownloadRemote /NoTipInfo /NoCheckFile .
)

setlocal

:: 获取当前用户名
set "USERPROFILE=%USERPROFILE%"

:: 目标目录
set "DEST_DIR=%USERPROFILE%\.cache\torch\hub\checkpoints"

:: 确保目标目录存在
if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"

:: 拷贝文件
copy /Y "doclayout\models.cache\resnet18-f37072fd.pth" "%DEST_DIR%"

echo 文件已拷贝到 %DEST_DIR%
endlocal

doclayout.exe
