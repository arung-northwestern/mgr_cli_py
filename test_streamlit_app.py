#!/usr/bin/env python3
"""Automated test for the Streamlit app using Playwright."""

import subprocess
import sys
import time
import traceback
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent
    app_path = repo_root / "python_cli" / "mgrcli" / "interactive_app.py"

    # Start Streamlit in the background
    env = {
        **subprocess.os.environ,
        "PYTHONPATH": str(repo_root) + ":" + str(repo_root / "mgr_src"),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true", "--server.port", "8502"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    # Wait for the server to start
    time.sleep(15)

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("http://localhost:8502")
            time.sleep(5)

            # Take a screenshot of initial state
            page.screenshot(path="streamlit_initial.png")
            print("Screenshot saved to streamlit_initial.png")

            # Try to select a sample CIF
            try:
                # Find the selectbox for sample CIFs
                page.click("text=Or select a sample CIF from the repository")
                time.sleep(1)
                # Try to select ZIF-8-RASPA.cif
                page.click("text=ZIF-8-RASPA.cif")
                time.sleep(3)
                page.screenshot(path="streamlit_after_select.png")
                print("Screenshot saved to streamlit_after_select.png")
            except Exception as e:
                print(f"Could not select sample CIF: {e}")
                # Try uploading instead
                file_input = page.locator('input[type="file"]').first
                if file_input:
                    file_input.set_input_files(str(repo_root / "cifs" / "ZIF-8-RASPA.cif"))
                    time.sleep(3)
                    page.screenshot(path="streamlit_after_upload.png")
                    print("Screenshot saved to streamlit_after_upload.png")

            # Click "Compute and Visualize"
            try:
                page.click("button:has-text('Compute and Visualize')")
                print("Clicked 'Compute and Visualize'")
            except Exception as e:
                print(f"Could not click button: {e}")
                # Try finding any button
                buttons = page.locator("button").all()
                for i, btn in enumerate(buttons):
                    print(f"Button {i}: {btn.inner_text()}")
                raise

            # Wait for computation
            time.sleep(20)
            page.screenshot(path="streamlit_after_compute.png")
            print("Screenshot saved to streamlit_after_compute.png")

            # Check for error messages
            error_elements = page.locator("[data-testid='stException']").all()
            if error_elements:
                print(f"\n*** FOUND {len(error_elements)} EXCEPTION(S) ***")
                for i, el in enumerate(error_elements):
                    print(f"\n--- Exception {i+1} ---")
                    print(el.inner_text())
            else:
                print("\nNo exceptions found in the UI.")

            # Check for status indicators
            body_text = page.locator("body").inner_text()
            if "COMPUTING" in body_text:
                print("Status: COMPUTING still visible")
            if "COMPUTE FINISHED" in body_text:
                print("Status: COMPUTE FINISHED visible")
            if "Plotting complete" in body_text:
                print("Status: Plotting complete visible")

            browser.close()
    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        stdout, stderr = proc.stdout.read(), proc.stderr.read()
        if stderr:
            print("\n--- Streamlit stderr ---")
            print(stderr[-5000:])  # Last 5000 chars
        if stdout:
            print("\n--- Streamlit stdout ---")
            print(stdout[-2000:])  # Last 2000 chars

if __name__ == "__main__":
    main()
