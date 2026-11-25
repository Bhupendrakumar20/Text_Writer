from fastapi.testclient import TestClient
from textwritter.web_server import app

client = TestClient(app)

def test_api():
    # 1. Info endpoint  
    r = client.get('/api/info')
    assert r.status_code == 200
    print('PASS: /api/info')
    # multilingual

    # 2. Multilingual Generate with long text
    long_text = 'The quick brown fox jumps over the lazy dog. Realistic student notebook writing wraps naturally across lines without truncation.'
    r2 = client.post('/api/multilingual/generate', data={'text': long_text, 'language': 'en', 'font_family': 'auto'})
    assert r2.status_code == 200
    data = r2.json()  
    
    assert data['success'] is True, f"Error: {data.get('error')}"
    print(f"PASS: /api/multilingual/generate -> {data['image_url']}, font={data.get('font_family')}")

    # 3. Notebook Generate
    r3 = client.post('/api/generate', data={'text': 'Test notebook text line 1\nLine 2 is here.', 'engine': 'notebook', 'notebook_style': 'kalam'})
    assert r3.status_code == 200
    data3 = r3.json()
    assert data3['success'] is True, f"Error: {data3.get('error')}"
    print(f"PASS: /api/generate notebook -> {data3['image_url']}")

    # 4. Upload print sample and verify style matching
    with open('assets/sample_styles/00_PRESIDENT.png', 'rb') as f:
        r4 = client.post(
            '/api/multilingual/generate',
            data={'text': 'Testing style matching from sample input image.', 'language': 'en', 'font_family': 'auto'},
            files=[('style_files', ('00_PRESIDENT.png', f, 'image/png'))]
        )
    assert r4.status_code == 200
    data4 = r4.json()
    assert data4['success'] is True
    print(f"PASS: Sample 00_PRESIDENT matched font={data4['font_family']} ({data4.get('font_family_name')})")
    assert data4['font_family'] == 'patrick', f"Expected patrick, got {data4['font_family']}"

    print('ALL INTEGRATION TESTS PASSED!')

if __name__ == '__main__':
    test_api()
