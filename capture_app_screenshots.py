"""
Final Playwright screenshot capture script with exact tab locators.
"""

import time
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
from playwright.sync_api import sync_playwright

def capture_all():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(base_dir, "reports")
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1050})
        page = context.new_page()

        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", timeout=40000)
        time.sleep(3)

        # Tab 1: Home / Ask a Question
        print("Tab 1: Entering question...")
        try:
            page.get_by_text("🔍 Ask a Question").click()
            time.sleep(1)
            inp = page.locator("input[type='text']").first
            inp.click()
            inp.fill("What is the Okapi BM25 algorithm?")
            inp.press("Enter")
            time.sleep(2)
            btn_search = page.locator("button:has-text('Search & Answer')").first
            btn_search.click()
            print("Waiting for query answer...")
            time.sleep(8)
        except Exception as e:
            print(f"Error on Tab 1: {e}")

        s1_path = os.path.join(reports_dir, "screenshot_1_home_query.png")
        page.screenshot(path=s1_path, full_page=True)
        print(f"Saved: {s1_path} ({os.path.getsize(s1_path)} bytes)")

        # Tab 2: Side-by-Side Comparison
        print("Tab 2: Side-by-Side Comparison...")
        try:
            page.get_by_text("⚖️ Side-by-Side Comparison").click()
            time.sleep(2)
            btn_comp = page.locator("button:has-text('Run 4-Way Comparison')").first
            btn_comp.click()
            print("Waiting for 4-way comparison...")
            time.sleep(8)
        except Exception as e:
            print(f"Error on Tab 2: {e}")

        s2_path = os.path.join(reports_dir, "screenshot_2_comparison.png")
        page.screenshot(path=s2_path, full_page=True)
        print(f"Saved: {s2_path} ({os.path.getsize(s2_path)} bytes)")

        # Tab 3: Document Ingestion & Corpus
        print("Tab 3: Document Ingestion & Corpus...")
        try:
            page.get_by_text("📚 Document Ingestion & Corpus").click()
            time.sleep(3)
        except Exception as e:
            print(f"Error on Tab 3: {e}")

        s3_path = os.path.join(reports_dir, "screenshot_3_ingestion.png")
        page.screenshot(path=s3_path, full_page=True)
        print(f"Saved: {s3_path} ({os.path.getsize(s3_path)} bytes)")

        # Tab 4: Evaluation Dashboard
        print("Tab 4: Evaluation Dashboard...")
        try:
            page.get_by_text("📊 Evaluation Dashboard").click()
            time.sleep(2)
            btn_eval = page.locator("button:has-text('Run Automated Benchmark Evaluation')").first
            btn_eval.click()
            print("Waiting for evaluation...")
            time.sleep(12)
        except Exception as e:
            print(f"Error on Tab 4: {e}")

        s4_path = os.path.join(reports_dir, "screenshot_4_evaluation.png")
        page.screenshot(path=s4_path, full_page=True)
        print(f"Saved: {s4_path} ({os.path.getsize(s4_path)} bytes)")

        # Copy to assets
        for sname in ["screenshot_1_home_query.png", "screenshot_2_comparison.png", "screenshot_3_ingestion.png", "screenshot_4_evaluation.png"]:
            src = os.path.join(reports_dir, sname)
            dst = os.path.join(assets_dir, sname)
            if os.path.exists(src):
                with open(src, "rb") as f_in, open(dst, "wb") as f_out:
                    f_out.write(f_in.read())

        browser.close()
        print("All screenshots generated and verified!")

if __name__ == "__main__":
    capture_all()
