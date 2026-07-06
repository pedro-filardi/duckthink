#!/usr/bin/env python3
# Proxy local HTTP->HTTPS para usar OpenAI desde duckthink (build sin TLS).
# Arranca:  python3 scripts/openai_proxy.py      (escucha en 127.0.0.1:8788)
# Lee la API key de ~/.openai_key
import http.server, socketserver, urllib.request, urllib.error, os, json, ssl, time
try:
    import certifi; CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    CTX = ssl._create_unverified_context()
KEY = open(os.path.expanduser('~/.openai_key')).read().strip()
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0)); body = self.rfile.read(n)
        data, code = b'{}', 502
        for attempt in range(8):
            req = urllib.request.Request('https://api.openai.com' + self.path, data=body,
                headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'}, method='POST')
            try:
                with urllib.request.urlopen(req, timeout=120, context=CTX) as r:
                    data, code = r.read(), r.status; break
            except urllib.error.HTTPError as e:
                code, data = e.code, e.read()
                if code == 429 or code >= 500:
                    ra = e.headers.get('Retry-After'); time.sleep(float(ra) if ra else min(1.5*(attempt+1), 8)); continue
                break
            except Exception as e:
                data = json.dumps({'error': {'message': str(e)}}).encode(); code = 502; time.sleep(1); continue
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
class T(socketserver.ThreadingMixIn, http.server.HTTPServer): daemon_threads = True
print('duckthink OpenAI proxy en http://127.0.0.1:8788  (Ctrl+C para parar)')
T(('127.0.0.1', 8788), H).serve_forever()
