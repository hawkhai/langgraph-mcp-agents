#encoding=utf8
import re, os, sys
#from app_tkinter import main
#from mcp_server_time import main
#from process_manager_fastmcp import main

if __name__ == "__main__":
    if "app" in sys.argv:
        from app_tkinter import main
        main()
   
    elif "mcp_time" in sys.argv:
        from mcp_server_time import main
        main()
        
    elif "mcp_proc" in sys.argv:
        from process_manager_fastmcp import main
        main()
        
    else:
        from app_tkinter import main
        main()
