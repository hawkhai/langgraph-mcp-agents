IF NOT EXIST "GraphAgentDesktop\GraphAgentDesktop.exe" (
    tar -xf GraphAgentDesktop.zip
)

gitbigfileftp.exe /DownloadRemote /NoTipInfo /NoCheckFile .
IF %ERRORLEVEL% EQU 0 (
    echo ����ɹ�ִ�У�����ֵΪ 0
) ELSE (
    echo ����ִ�г������⣬����ֵΪ %ERRORLEVEL%
    timeout /T 5 /NOBREAK >nul
    gitbigfileftp.exe /DownloadRemote /NoTipInfo /NoCheckFile .
)
