#!/usr/bin/env python3
"""
TrafficTwin Project Launcher for VS Code
Usage:
  python run.py         -> Starts local visualizer server and opens browser
  python run.py --cli   -> Runs Python calibration benchmark
"""

import sys
import webbrowser
import http.server
import socketserver
import os

PORT = 8080

def start_web_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        url = f"http://localhost:{PORT}/"
        print(f"\n[+] TrafficTwin Visualizer running at: {url}")
        print("[+] Opening default browser...")
        print("[+] Press Ctrl+C in terminal to stop the server.\n")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        import calibrate
        calibrate.main()
    else:
        start_web_server()
