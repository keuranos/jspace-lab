#!/usr/bin/env python3
"""jspace_live_view.py — browser dashboard for jspace_live.py JSONL logs.

Serves an HTML page (no external deps) that polls /api/data and renders:
  - channel A: emitted token stream (what the mouth says)
  - per-token engagement axis strip (green=engagement, red=deflection)
  - VETO badges (deepest-layer top-1 differs from emitted)
  - concept trajectories: yes/no/conscious/... probability across L48-62
  - click a token -> its full per-layer top-8 readout

Usage on your GPU host:
  python3 jspace_live_view.py [--port 11450] [--dir ~/jspace-precision]
Then open http://<host-ip>:11450/
"""
import argparse, json, os, glob
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>j-space live</title>
<style>
  body{background:#0b0e14;color:#cdd6f4;font-family:ui-monospace,Menlo,Consolas,monospace;margin:16px}
  h1{font-size:15px;color:#89b4fa;margin:0 0 4px}
  #meta{color:#6c7086;font-size:11px;margin-bottom:10px}
  #filebar{margin-bottom:8px}select{background:#11111b;color:#cdd6f4;border:1px solid #313244;padding:2px}
  .tok{padding:1px 2px;border-radius:3px;cursor:pointer;white-space:pre-wrap}
  .tok.veto{outline:1px solid #f9e2af;background:#3a3016}
  .tok.eng{color:#a6e3a1}.tok.def{color:#f38ba8}.tok.neu{color:#cdd6f4}
  #stream{font-size:15px;line-height:1.55;border:1px solid #313244;padding:10px;border-radius:6px;min-height:60px;background:#11111b}
  .row{margin:14px 0}.lbl{font-size:11px;color:#6c7086;margin-bottom:3px}
  canvas{width:100%;height:120px;background:#11111b;border:1px solid #313244;border-radius:6px}
  #detail{font-size:11px;border:1px solid #45475a;padding:8px;border-radius:6px;background:#11111b;white-space:pre;display:none;max-height:280px;overflow:auto}
  .sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:-1px}
</style></head><body>
<h1>j-space live viewer — Qwen3.8-27B + Jacobian lens</h1>
<div id="meta">loading…</div>
<div id="filebar"><span class="lbl">log:</span> <select id="files"></select></div>
<div class="row"><div class="lbl">CHANNEL A — emitted tokens (hover: none, click: full readout). Yellow outline = VETO (L62 wanted something else). Green/red fill = engagement/deflection axis.</div>
<div id="stream"></div></div>
<div class="row"><div class="lbl">ENGAGEMENT AXIS per token (−1 deflection … +1 engagement)</div>
<canvas id="axis" width="1200" height="120"></canvas></div>
<div class="row"><div class="lbl">CONCEPT MASS at L55 across generation — <span class="sw" style="background:#a6e3a1"></span>yes <span class="sw" style="background:#f38ba8"></span>no <span class="sw" style="background:#89b4fa"></span>conscious <span class="sw" style="background:#f9e2af"></span>experience <span class="sw" style="background:#cba6f7"></span>feel</div>
<canvas id="concepts" width="1200" height="120"></canvas></div>
<div class="row"><div class="lbl">PER-LAYER READOUT (click a token above)</div><div id="detail"></div></div>
<script>
let DATA=null, SEL=-1;
const CKEYS=["yes","no","conscious","experience","feel"];
const COLORS={yes:"#a6e3a1",no:"#f38ba8",conscious:"#89b4fa",experience:"#f9e2af",feel:"#cba6f7"};
function cls(r){return r.veto?"veto":(r.engagement_axis>0.15?"eng":r.engagement_axis<-0.15?"def":"neu");}
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
async function refresh(){
  const f=document.getElementById("files").value;
  const r=await fetch("/api/data?file="+encodeURIComponent(f));
  DATA=await r.json();
  document.getElementById("meta").textContent=
    DATA.length+" tokens | veto flags: "+DATA.filter(x=>x.veto).length+
    " | model: Qwen3.8-27B NF4 cuda:1 | lens L48-62, veto ref L62";
  const st=document.getElementById("stream");
  st.innerHTML=DATA.map((t,i)=>
    `<span class="tok ${cls(t)}" onclick="show(${i})" title="axis ${t.engagement_axis}${t.veto?" VETO":""}">${esc(t.emitted_token)}</span>`).join("");
  drawAxis();drawConcepts();if(SEL>=0)show(SEL);
}
function drawAxis(){
  const c=document.getElementById("axis"),x=c.getContext("2d");
  x.clearRect(0,0,c.width,c.height);if(!DATA.length)return;
  const W=c.width,H=c.height,mid=H/2,n=DATA.length;
  x.strokeStyle="#313244";x.beginPath();x.moveTo(0,mid);x.lineTo(W,mid);x.stroke();
  const bw=Math.max(2,W/n);
  DATA.forEach((t,i)=>{const v=Math.max(-1,Math.min(1,t.engagement_axis));
    x.fillStyle=v>=0?"#a6e3a1":"#f38ba8";
    const h=Math.abs(v)*(H/2-4);
    x.fillRect(i*bw, v>=0?mid-h:mid, Math.max(1,bw-1), Math.max(1,h));});
}
function drawConcepts(){
  const c=document.getElementById("concepts"),x=c.getContext("2d");
  x.clearRect(0,0,c.width,c.height);if(!DATA.length)return;
  const W=c.width,H=c.height,n=DATA.length;
  x.strokeStyle="#313244";[0.25,0.5,0.75].forEach(f=>{x.beginPath();x.moveTo(0,H*f);x.lineTo(W,H*f);x.stroke();});
  // per-concept max normalization (raw masses can be ~0.001; shape matters)
  const mx={};CKEYS.forEach(k=>{mx[k]=1e-9;
    DATA.forEach(t=>{const v=(t.concept_mass||{})[k]||0;if(v>mx[k])mx[k]=v;});});
  CKEYS.forEach(k=>{
    x.strokeStyle=COLORS[k];x.lineWidth=1.5;x.beginPath();let started=false;
    DATA.forEach((t,i)=>{const v=((t.concept_mass||{})[k]||0)/mx[k];
      const px=i*(W/n)+1, py=H-4-v*(H-10);
      if(!started){x.moveTo(px,py);started=true;}else x.lineTo(px,py);});
    x.stroke();});
  // max labels
  x.fillStyle="#6c7086";x.font="10px monospace";
  CKEYS.forEach((k,j)=>{
    x.fillText("max "+k+"="+(mx[k]<0.01?mx[k].toExponential(1):mx[k].toFixed(3)), 8+j*130, 12);});
}
function show(i){
  SEL=i;const t=DATA[i],d=document.getElementById("detail");
  const lay=t.layers||{};
  let out=`step ${t.step}  emitted ${JSON.stringify(t.emitted_token)}  axis ${t.engagement_axis}  veto=${t.veto}\\n\\n`;
  out+="final top: "+(t.final_topk||[]).map(p=>p[0]+" "+p[1].toFixed(2)).join(" | ")+"\\n\\n";
  Object.keys(lay).map(Number).sort((a,b)=>a-b).forEach(l=>{
    out+="L"+String(l).padStart(2)+":  "+lay[l].map(p=>p[0]+" "+p[1].toFixed(3)).join("  ")+"\\n";});
  d.textContent=out;d.style.display="block";
}
async function init(){
  const r=await fetch("/api/list");const files=await r.json();
  const sel=document.getElementById("files");
  sel.innerHTML=files.map(f=>`<option>${f}</option>`).join("");
  sel.onchange=()=>{SEL=-1;document.getElementById("detail").style.display="none";refresh();};
  if(files.length)refresh();else document.getElementById("meta").textContent="no live_*.jsonl logs found";
}
init();setInterval(()=>{const s=SEL;refresh().then(()=>{SEL=s;});},2000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *a, logdir=None, **kw):
        self.logdir = logdir
        super().__init__(*a, **kw)

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return
        if self.path == "/api/list":
            files = sorted((os.path.basename(p) for p in
                            glob.glob(os.path.join(self.logdir, "live_*.jsonl"))),
                           reverse=True)
            self._send(200, json.dumps(files).encode(), "application/json")
            return
        if self.path.startswith("/api/data"):
            from urllib.parse import urlparse, parse_qs, unquote
            q = parse_qs(urlparse(self.path).query)
            name = os.path.basename(unquote(q.get("file", [""])[0]))
            path = os.path.join(self.logdir, name)
            recs = []
            if os.path.isfile(path):
                with open(path) as f:
                    recs = [json.loads(line) for line in f if line.strip()]
            self._send(200, json.dumps(recs).encode(), "application/json")
            return
        self._send(404, b"not found", "text/plain")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11450)
    ap.add_argument("--dir", default=os.path.expanduser("~/jspace-precision"))
    args = ap.parse_args()

    handler = lambda *a, **kw: Handler(*a, logdir=args.dir, **kw)
    print(f"[live-view] serving {args.dir}/live_*.jsonl on 0.0.0.0:{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), handler).serve_forever()
