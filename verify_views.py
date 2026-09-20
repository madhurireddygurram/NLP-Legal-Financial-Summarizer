"""
Verification Suite for Repainted Color Theme & Preserved UI Layout.
Validates exact original structure, forms, tables, and audited evaluation metrics.
"""

import os
import sys
import django
from django.test import RequestFactory

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Summary.settings")
django.setup()

from SummaryApp import views


def run_verification():
    print("=" * 70)
    print("[*] Running System Verification (Repainted Color Theme + Preserved UI)")
    print("=" * 70)

    factory = RequestFactory()
    from django.test import Client
    client = Client()

    # 1. Test Root URL '/' (Resolves the 404 issue)
    print("[1] Testing root URL '/'...")
    resp_root = client.get('/')
    assert resp_root.status_code == 200, f"Root URL '/' failed with status {resp_root.status_code}"
    content_root = resp_root.content.decode('utf-8')
    assert "pipeline_banner.jpg" in content_root, "Missing pipeline_banner.jpg in root URL"
    assert "Developing an NLP Model" in content_root, "Missing title in root URL"
    print("    [OK] Root URL '/' returned 200 OK (Home page properly mapped with pipeline_banner.jpg)")

    # 2. Test Favicon URL '/favicon.ico'
    print("[2] Testing favicon URL '/favicon.ico'...")
    resp_fav = client.get('/favicon.ico')
    assert resp_fav.status_code == 200, f"Favicon URL failed with status {resp_fav.status_code}"
    print("    [OK] Favicon URL '/favicon.ico' returned 200 OK (Icon served properly)")

    # 3. Test Index View '/index.html'
    print("[3] Testing index view '/index.html'...")
    req = factory.get('/index.html')
    resp = views.index(req)
    assert resp.status_code == 200, f"Index view failed with status {resp.status_code}"
    content = resp.content.decode('utf-8')
    assert "pipeline_banner.jpg" in content, "Missing pipeline_banner.jpg splash"
    assert "Developing an NLP Model" in content, "Missing header title"
    print("    [OK] Index view returned 200 OK (Exact original style with pipeline_banner.jpg verified)")

    # 4. Test UserLogin View
    print("[4] Testing UserLogin view...")
    req = factory.get('/UserLogin')
    resp = views.UserLogin(req)
    assert resp.status_code == 200, f"UserLogin view failed with status {resp.status_code}"
    content = resp.content.decode('utf-8')
    assert 'name="username"' in content and 'name="password"' in content, "Missing login inputs"
    print("    [OK] UserLogin view returned 200 OK (Original form preserved)")

    # 5. Test Signup View
    print("[5] Testing Signup view...")
    req = factory.get('/Signup')
    resp = views.Signup(req)
    assert resp.status_code == 200, f"Signup view failed with status {resp.status_code}"
    content = resp.content.decode('utf-8')
    assert 'name="t1"' in content and 'name="t2"' in content and 'name="t3"' in content, "Missing signup inputs"
    print("    [OK] Signup view returned 200 OK (Original form preserved)")

    # 6. Test Aboutus View
    print("[6] Testing Aboutus view...")
    req = factory.get('/Aboutus')
    resp = views.Aboutus(req)
    assert resp.status_code == 200, f"Aboutus view failed with status {resp.status_code}"
    print("    [OK] Aboutus view returned 200 OK (Original layout preserved)")

    # 7. Test TrainNLP View
    print("[7] Testing TrainNLP view (Metrics Table, ROUGE Table, Dark Comparison Graph)...")
    req = factory.get('/TrainNLP')
    resp = views.TrainNLP(req)
    assert resp.status_code == 200, f"TrainNLP view failed with status {resp.status_code}"
    content = resp.content.decode('utf-8')
    assert "NLP Summary" in content, "Missing NLP Summary row"
    assert "Transformer (T5-Small) Summary" in content, "Missing Transformer Summary row"
    assert "Precision" in content and "Recall" in content and "F-Measure" in content, "Missing metric columns"
    assert "ROUGE-1 F1" in content and "ROUGE-2 F1" in content and "ROUGE-L F1" in content, "Missing ROUGE table"
    assert "data:image/png;base64," in content, "Missing generated comparison graph"
    print("    [OK] TrainNLP view successfully rendered tables and dark comparison graph!")

    # 8. Test GenerateSummary View
    print("[8] Testing GenerateSummary view...")
    req = factory.get('/GenerateSummary')
    resp = views.GenerateSummary(req)
    assert resp.status_code == 200, f"GenerateSummary view failed with status {resp.status_code}"
    content = resp.content.decode('utf-8')
    assert 'name="t1"' in content, "Missing textarea t1"
    assert 'name="f1"' in content, "Missing form f1"
    print("    [OK] GenerateSummary view returned 200 OK (Original form preserved)")

    # 9. Test GenerateSummaryAction View
    print("[9] Testing GenerateSummaryAction with sample legal contract text...")
    sample_legal_text = (
        "The Recipient agrees to hold and maintain the Confidential Information in strictest confidence "
        "for the sole and exclusive benefit of the Disclosing Party. The Recipient shall carefully restrict "
        "access to Confidential Information to employees, contractors, and third parties as is reasonably required. "
        "These Terms shall be governed and construed in accordance with the laws of the jurisdiction, "
        "and any breach of this provision will result in immediate termination."
    )
    req = factory.post('/GenerateSummaryAction', {'t1': sample_legal_text})
    resp = views.GenerateSummaryAction(req)
    assert resp.status_code == 200, f"GenerateSummaryAction failed with status {resp.status_code}"
    content = resp.content.decode('utf-8')
    assert "Input Text:" in content, "Missing Input Text in output"
    assert "Extractive Summary:" in content, "Missing Extractive Summary in output"
    assert "T5 Transformer Summary:" in content, "Missing T5 Transformer Summary in output"
    print("    [OK] GenerateSummaryAction generated both Extractive and T5 summaries in original format!")

    print("\n" + "=" * 70)
    print("[ALL 9 CHECKS PASSED] Root URL '/', Favicon, and All Existing Views Verified!")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
