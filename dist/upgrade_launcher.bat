IF NOT EXIST "doclayout\doclayout.exe" (
    tar -xf doclayout.zip
)

gitbigfileftp.exe /DownloadRemote /NoTipInfo /NoCheckFile .
IF %ERRORLEVEL% EQU 0 (
    echo ����ɹ�ִ�У�����ֵΪ 0
) ELSE (
    echo ����ִ�г������⣬����ֵΪ %ERRORLEVEL%
    timeout /T 5 /NOBREAK >nul
    gitbigfileftp.exe /DownloadRemote /NoTipInfo /NoCheckFile .
)

setlocal

:: ��ȡ��ǰ�û���
set "USERPROFILE=%USERPROFILE%"

:: Ŀ��Ŀ¼
set "DEST_DIR=%USERPROFILE%\.cache\torch\hub\checkpoints"

:: ȷ��Ŀ��Ŀ¼����
if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"

:: �����ļ�
copy /Y "doclayout\models.cache\resnet18-f37072fd.pth" "%DEST_DIR%"

echo �ļ��ѿ����� %DEST_DIR%
endlocal

doclayout.exe
