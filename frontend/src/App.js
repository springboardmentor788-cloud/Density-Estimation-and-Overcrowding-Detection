import { useState, useRef, useCallback, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, AreaChart, Area
} from "recharts";
import {
  Upload, BarChart3, Home, AlertTriangle, CheckCircle,
  Download, Moon, Sun, Users, Activity, Zap,
  ChevronRight, X, Eye, TrendingUp, Database,
  Bell, Shield, Play, Image as ImageIcon, Camera,
  Maximize2, RefreshCw
} from "lucide-react";

// ─────────────────────────────────────────────────────────────────────────────
//  SAMPLE DATASET IMAGES  (from ShanghaiTech Part A/B style crowds)
//  These are real crowd image URLs from Unsplash/Pexels for demo purposes.
//  In production these would be your actual dataset files from:
//  dataset/archive/part_A_final/test_data/images/
// ─────────────────────────────────────────────────────────────────────────────
const SAMPLE_IMAGES = [
  { id: 1, name: "IMG_001.jpg", label: "Part A — Dense",   url: "https://images.unsplash.com/photo-1522158637959-30385a09e0da?w=400&q=80", count: 847 },
  { id: 2, name: "IMG_002.jpg", label: "Part A — Festival",url: "https://images.unsplash.com/photo-1429962714451-bb934ecdc4ec?w=400&q=80", count: 1243 },
  { id: 3, name: "IMG_003.jpg", label: "Part B — Street",  url: "https://images.unsplash.com/photo-1514924013411-cbf25faa35bb?w=400&q=80", count: 312 },
  { id: 4, name: "IMG_004.jpg", label: "Part B — Station", url: "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=400&q=80", count: 198 },
  { id: 5, name: "IMG_005.jpg", label: "Part A — Concert", url: "https://images.unsplash.com/photo-1501281668745-f7f57925c3b4?w=400&q=80", count: 965 },
  { id: 6, name: "IMG_006.jpg", label: "Part B — Market",  url: "https://images.unsplash.com/photo-1469371670807-013ccf25f16a?w=400&q=80", count: 156 },
];

// ─────────────────────────────────────────────────────────────────────────────
//  MOCK API — replace with real axios calls to your FastAPI backend
// ─────────────────────────────────────────────────────────────────────────────
// import axios from 'axios';
// const BASE = 'http://localhost:8000';
// Real: await axios.post(`${BASE}/upload`, formData, { onUploadProgress })
// Real: await axios.get(`${BASE}/results`)

const API = {
  analyse: (file, threshold, onProgress, predefinedCount) =>
    new Promise((resolve) => {
      let p = 0;
      const iv = setInterval(() => {
        p += Math.random() * 14 + 6;
        if (p >= 100) { clearInterval(iv); p = 100; }
        onProgress(Math.min(p, 100));
        if (p >= 100) {
          const count = predefinedCount || Math.floor(Math.random() * 900 + 60);
          const frames = file?.type?.startsWith("video") ? Math.floor(Math.random() * 120 + 20) : 1;
          resolve({
            count,
            frames,
            avg: count,
            peak: count,
            fps: file?.type?.startsWith("video") ? Math.floor(Math.random() * 25 + 5) : 0,
            alert: count >= threshold,
            processing_time: (Math.random() * 3 + 0.8).toFixed(2) + "s",
            scale_factor: (Math.random() * 2 + 1.2).toFixed(4),
            model: "CSRNet Pretrained",
            time: new Date().toLocaleTimeString(),
            isVideo: file?.type?.startsWith("video") || false,
          });
        }
      }, 160);
    }),

  getHistory: () =>
    Promise.resolve(
      Array.from({ length: 8 }, (_, i) => ({
        id: i + 1,
        file: `crowd_${i + 1}.jpg`,
        count: Math.floor(Math.random() * 900 + 80),
        time: `${String(Math.floor(Math.random() * 24)).padStart(2,"0")}:${String(Math.floor(Math.random()*60)).padStart(2,"0")}`,
        date: `2026-04-${String(i+1).padStart(2,"0")}`,
        threshold: 300,
        alert: Math.random() > 0.55,
      }))
    ),
};

const C = {
  accent:  "#00E5CC",
  accent2: "#FF6B35",
  warn:    "#FFB347",
  danger:  "#FF4757",
  success: "#2ED573",
  blue:    "#4A9EFF",
};

// ─────────────────────────────────────────────────────────────────────────────
//  DENSITY HEATMAP SVG
// ─────────────────────────────────────────────────────────────────────────────
function DensityHeatmap({ count, threshold, height = 200, style = {} }) {
  const intensity = Math.min(count / 1400, 1);
  const isAlert   = count >= threshold;
  const hot       = isAlert ? "#ef4444" : "#f97316";
  return (
    <div style={{ position:"relative", borderRadius:8, overflow:"hidden", height, background:"#020818", ...style }}>
      <svg viewBox="0 0 400 200" preserveAspectRatio="none"
        style={{ width:"100%", height:"100%", position:"absolute", inset:0 }}>
        <defs>
          <radialGradient id="h1" cx="50%" cy="48%"><stop offset="0%" stopColor={hot}     stopOpacity={0.95*intensity}/><stop offset="100%" stopColor="transparent"/></radialGradient>
          <radialGradient id="h2" cx="27%" cy="32%"><stop offset="0%" stopColor="#f97316" stopOpacity={0.80*intensity}/><stop offset="100%" stopColor="transparent"/></radialGradient>
          <radialGradient id="h3" cx="74%" cy="65%"><stop offset="0%" stopColor="#a855f7" stopOpacity={0.70*intensity}/><stop offset="100%" stopColor="transparent"/></radialGradient>
          <radialGradient id="h4" cx="16%" cy="76%"><stop offset="0%" stopColor="#3b82f6" stopOpacity={0.55*intensity}/><stop offset="100%" stopColor="transparent"/></radialGradient>
          <radialGradient id="h5" cx="83%" cy="25%"><stop offset="0%" stopColor="#22c55e" stopOpacity={0.42*intensity}/><stop offset="100%" stopColor="transparent"/></radialGradient>
          <radialGradient id="h6" cx="60%" cy="80%"><stop offset="0%" stopColor="#06b6d4" stopOpacity={0.35*intensity}/><stop offset="100%" stopColor="transparent"/></radialGradient>
        </defs>
        {/* base cold blue background */}
        <rect width="400" height="200" fill="#0a1628" opacity="0.9"/>
        <ellipse cx="200" cy="96"  rx="160" ry="100" fill="url(#h1)"/>
        <ellipse cx="108" cy="64"  rx="100" ry="72"  fill="url(#h2)"/>
        <ellipse cx="296" cy="130" rx="110" ry="78"  fill="url(#h3)"/>
        <ellipse cx="64"  cy="152" rx="70"  ry="52"  fill="url(#h4)"/>
        <ellipse cx="340" cy="50"  rx="80"  ry="60"  fill="url(#h5)"/>
        <ellipse cx="240" cy="170" rx="65"  ry="45"  fill="url(#h6)"/>
      </svg>
      {/* legend */}
      <div style={{ position:"absolute", bottom:8, right:10, display:"flex", alignItems:"center", gap:4, fontSize:10, color:"rgba(255,255,255,.5)" }}>
        <span>Low</span>
        {["#3b82f6","#06b6d4","#22c55e","#a855f7","#f97316",hot].map((c,i)=>(
          <div key={i} style={{ width:12, height:6, background:c, borderRadius:2 }}/>
        ))}
        <span>High</span>
      </div>
      {isAlert && (
        <div style={{ position:"absolute", inset:0, border:"2px solid #ef4444", borderRadius:8, pointerEvents:"none", animation:"alertBorder 1.4s ease-in-out infinite" }}/>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  OVERLAY DENSITY (coloured version of the input — simulated)
// ─────────────────────────────────────────────────────────────────────────────
function OverlayDensity({ previewSrc, count, height = 200 }) {
  const intensity = Math.min(count / 1400, 1);
  return (
    <div style={{ position:"relative", height, background:"#020818", borderRadius:8, overflow:"hidden" }}>
      {previewSrc && (
        <img src={previewSrc} alt="overlay" style={{ width:"100%", height:"100%", objectFit:"cover", display:"block", opacity:0.55 }}/>
      )}
      {/* colour overlay */}
      <svg viewBox="0 0 400 200" preserveAspectRatio="none"
        style={{ position:"absolute", inset:0, width:"100%", height:"100%", mixBlendMode:"screen" }}>
        <defs>
          <radialGradient id="ov1" cx="50%" cy="48%"><stop offset="0%" stopColor="#ff4500" stopOpacity={0.85*intensity}/><stop offset="100%" stopColor="transparent"/></radialGradient>
          <radialGradient id="ov2" cx="27%" cy="32%"><stop offset="0%" stopColor="#ff8c00" stopOpacity={0.65*intensity}/><stop offset="100%" stopColor="transparent"/></radialGradient>
          <radialGradient id="ov3" cx="74%" cy="65%"><stop offset="0%" stopColor="#ffd700" stopOpacity={0.55*intensity}/><stop offset="100%" stopColor="transparent"/></radialGradient>
          <radialGradient id="ov4" cx="16%" cy="76%"><stop offset="0%" stopColor="#00ced1" stopOpacity={0.45*intensity}/><stop offset="100%" stopColor="transparent"/></radialGradient>
        </defs>
        <ellipse cx="200" cy="96"  rx="150" ry="95" fill="url(#ov1)"/>
        <ellipse cx="108" cy="64"  rx="90"  ry="65" fill="url(#ov2)"/>
        <ellipse cx="296" cy="130" rx="100" ry="70" fill="url(#ov3)"/>
        <ellipse cx="64"  cy="152" rx="65"  ry="48" fill="url(#ov4)"/>
      </svg>
      <div style={{ position:"absolute", top:8, left:8, fontSize:10, color:"rgba(255,255,255,.6)", background:"rgba(0,0,0,.5)", padding:"2px 8px", borderRadius:4, letterSpacing:"0.06em" }}>DENSITY OVERLAY</div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  LIVE STATS ROW  (Frames · Average · Peak Count · FPS)
// ─────────────────────────────────────────────────────────────────────────────
function StatsRow({ result, dark }) {
  const text  = dark ? "#e0e8ff" : "#1a1f35";
  const muted = dark ? "#5a6478" : "#94a3b8";
  const bg    = dark ? "#161b27" : "#fff";
  const bdr   = dark ? "#1e2535" : "#e2e8f8";

  const stats = [
    { icon: "⊞", label: "Frames",     value: result.frames,                      color: C.accent  },
    { icon: "⌀", label: "Average",    value: result.avg,                          color: C.blue    },
    { icon: "↑", label: "Peak Count", value: result.peak,                         color: C.warn    },
    { icon: "⚡", label: "FPS",        value: result.fps,                          color: C.success },
  ];

  return (
    <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:12, marginBottom:16 }}>
      {stats.map(({ icon, label, value, color }) => (
        <div key={label} style={{ background:bg, border:`1px solid ${bdr}`, borderRadius:12, padding:"14px 18px", position:"relative", overflow:"hidden" }}>
          <div style={{ position:"absolute", top:-16, right:-16, width:60, height:60, borderRadius:"50%", background:color+"10" }}/>
          <div style={{ fontSize:11, color:muted, marginBottom:4, display:"flex", alignItems:"center", gap:5 }}>
            <span style={{ fontSize:13 }}>{icon}</span>{label}
          </div>
          <div style={{ fontSize:30, fontWeight:800, color:color, fontFamily:"'Space Mono',monospace", lineHeight:1 }}>
            {value.toLocaleString()}
          </div>
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  ALERT BANNER  (matches screenshot exactly)
// ─────────────────────────────────────────────────────────────────────────────
function AlertBanner({ count, threshold, onDismiss }) {
  if (count < threshold) return null;
  return (
    <div style={{ background:"#1a0505", border:"1px solid #FF4757", borderLeft:"4px solid #FF4757", borderRadius:10, padding:"13px 18px", display:"flex", alignItems:"center", gap:12, marginBottom:16, animation:"slideDown 0.35s ease" }}>
      <AlertTriangle size={18} color="#FF4757" style={{ flexShrink:0, animation:"alertPulse 1.4s ease-in-out infinite" }}/>
      <span style={{ fontSize:14, color:"#FF4757", fontWeight:700 }}>Overcrowding Detected</span>
      <span style={{ fontSize:13, color:"#ff9999" }}>— Current count ({count.toLocaleString()}) exceeds threshold ({threshold.toLocaleString()})</span>
      {onDismiss && (
        <button onClick={onDismiss} style={{ marginLeft:"auto", background:"none", border:"none", color:"#ff9999", cursor:"pointer" }}><X size={14}/></button>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  THREE-PANEL OUTPUT  (INPUT · OVERLAY DENSITY · DENSITY MAP HEATMAP)
// ─────────────────────────────────────────────────────────────────────────────
function ThreePanelOutput({ result, preview, fileType, threshold, dark }) {
  const bg  = dark ? "#0d1117" : "#f0f4ff";
  const bdr = dark ? "#1e2535" : "#e2e8f8";
  const text= dark ? "#e0e8ff" : "#1a1f35";

  const panelStyle = {
    background:"#050a18",
    border:`1px solid ${bdr}`,
    borderRadius:10,
    overflow:"hidden",
    display:"flex",
    flexDirection:"column",
  };

  const headerStyle = (label, badge, badgeColor) => (
    <div style={{ padding:"8px 12px", borderBottom:`1px solid ${dark?"#1a2030":"#e2e8f8"}`, display:"flex", alignItems:"center", gap:8, background:dark?"#0d1117":"#f8faff" }}>
      <span style={{ fontSize:11, fontWeight:700, color:dark?"#8899bb":"#667", letterSpacing:"0.08em" }}>{label}</span>
      {badge && (
        <span style={{ fontSize:10, fontWeight:700, padding:"2px 8px", borderRadius:4, background:badgeColor+"25", color:badgeColor, border:`1px solid ${badgeColor}40`, letterSpacing:"0.06em" }}>{badge}</span>
      )}
    </div>
  );

  return (
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:12 }}>

      {/* Panel 1: INPUT */}
      <div style={panelStyle}>
        {headerStyle("INPUT", fileType === "video" ? "VIDEO" : "CAMERA 0", C.accent)}
        <div style={{ flex:1, position:"relative" }}>
          {preview ? (
            fileType === "video"
              ? <video src={preview} controls style={{ width:"100%", height:220, objectFit:"cover", display:"block" }}/>
              : <img src={preview} alt="input" style={{ width:"100%", height:220, objectFit:"cover", display:"block" }}/>
          ) : (
            <div style={{ height:220, display:"flex", alignItems:"center", justifyContent:"center", color:"#4a5568", fontSize:13, flexDirection:"column", gap:8 }}>
              <Camera size={28} style={{ opacity:.3 }}/>
              <span>No input</span>
            </div>
          )}
          {/* count overlay */}
          {result && (
            <div style={{ position:"absolute", bottom:10, left:10, background:"rgba(0,0,0,.75)", backdropFilter:"blur(4px)", borderRadius:8, padding:"6px 12px", display:"flex", alignItems:"center", gap:7, border:`1px solid ${result.alert?C.danger+"60":"rgba(255,255,255,.1)"}` }}>
              <Users size={14} color="#fff"/>
              <span style={{ fontSize:18, fontWeight:800, fontFamily:"'Space Mono',monospace", color: result.alert ? C.danger : C.accent }}>{result.count.toLocaleString()}</span>
              {result.alert && <AlertTriangle size={13} color={C.danger}/>}
            </div>
          )}
        </div>
      </div>

      {/* Panel 2: OVERLAY DENSITY */}
      <div style={panelStyle}>
        {headerStyle("OVERLAY", "DENSITY", C.accent2)}
        <div style={{ flex:1 }}>
          {result
            ? <OverlayDensity previewSrc={preview} count={result.count} height={220}/>
            : <div style={{ height:220, display:"flex", alignItems:"center", justifyContent:"center", color:"#4a5568", fontSize:13, flexDirection:"column", gap:8 }}><Activity size={28} style={{ opacity:.3 }}/><span>Run analysis first</span></div>
          }
        </div>
      </div>

      {/* Panel 3: DENSITY MAP HEATMAP */}
      <div style={panelStyle}>
        {headerStyle("DENSITY MAP", "HEATMAP", C.warn)}
        <div style={{ flex:1 }}>
          {result
            ? <DensityHeatmap count={result.count} threshold={threshold} height={220}/>
            : <div style={{ height:220, display:"flex", alignItems:"center", justifyContent:"center", color:"#4a5568", fontSize:13, flexDirection:"column", gap:8 }}><Eye size={28} style={{ opacity:.3 }}/><span>Run analysis first</span></div>
          }
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  CROWD COUNT LIVE BAR  (bottom bar like in screenshot)
// ─────────────────────────────────────────────────────────────────────────────
function CrowdCountBar({ result, threshold, dark }) {
  if (!result) return null;
  const pct = Math.min((result.count / (threshold * 2)) * 100, 100);
  const bdr = dark ? "#1e2535" : "#e2e8f8";
  return (
    <div style={{ background:dark?"#0d1117":"#f0f4ff", border:`1px solid ${bdr}`, borderRadius:10, padding:"12px 18px", marginTop:12, display:"flex", alignItems:"center", gap:16 }}>
      <div style={{ display:"flex", alignItems:"center", gap:8 }}>
        <div style={{ width:10, height:10, borderRadius:"50%", background:result.alert?C.danger:C.success, animation:result.alert?"alertPulse 1s infinite":"none", boxShadow:`0 0 6px ${result.alert?C.danger:C.success}` }}/>
        <span style={{ fontSize:12, fontWeight:700, color:dark?"#8899bb":"#667", letterSpacing:"0.06em" }}>CROWD COUNT — LIVE</span>
      </div>
      <div style={{ flex:1, height:6, background:dark?"#1e2535":"#e2e8f8", borderRadius:99, overflow:"hidden" }}>
        <div style={{ height:"100%", width:`${pct}%`, background:result.alert?`linear-gradient(90deg,${C.warn},${C.danger})`:`linear-gradient(90deg,${C.success},${C.accent})`, borderRadius:99, transition:"width 0.6s ease" }}/>
      </div>
      <span style={{ fontSize:16, fontWeight:800, fontFamily:"'Space Mono',monospace", color:result.alert?C.danger:C.accent, minWidth:60, textAlign:"right" }}>{result.count.toLocaleString()}</span>
      <span style={{ fontSize:12, color:dark?"#5a6478":"#94a3b8" }}>/ {threshold} threshold</span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  SAMPLE IMAGES PANEL
// ─────────────────────────────────────────────────────────────────────────────
function SamplePanel({ onSelect, dark }) {
  const bdr  = dark ? "#1e2535" : "#e2e8f8";
  const text = dark ? "#e0e8ff" : "#1a1f35";
  const muted= dark ? "#5a6478" : "#94a3b8";
  const bg   = dark ? "#161b27" : "#fff";
  const [dragOver, setDragOver] = useState(null);

  return (
    <div style={{ background:bg, border:`1px solid ${bdr}`, borderRadius:14, overflow:"hidden" }}>
      <div style={{ padding:"14px 18px", borderBottom:`1px solid ${bdr}`, display:"flex", alignItems:"center", gap:8 }}>
        <Database size={15} color={C.accent}/>
        <span style={{ fontSize:13, fontWeight:700, color:text }}>Sample Dataset Images</span>
        <span style={{ fontSize:11, color:muted, marginLeft:4 }}>ShanghaiTech Part A &amp; B — drag or click to analyse</span>
      </div>
      <div style={{ padding:14, display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:10 }}>
        {SAMPLE_IMAGES.map(img => (
          <div key={img.id}
            draggable
            onDragStart={e => { e.dataTransfer.setData("sampleId", img.id); }}
            onDragOver={e => { e.preventDefault(); setDragOver(img.id); }}
            onDragLeave={() => setDragOver(null)}
            onClick={() => onSelect(img)}
            style={{ borderRadius:10, overflow:"hidden", cursor:"pointer", border:`2px solid ${dragOver===img.id?C.accent:bdr}`, transition:"all 0.2s", transform:dragOver===img.id?"scale(1.03)":"scale(1)" }}>
            <div style={{ position:"relative" }}>
              <img src={img.url} alt={img.name} style={{ width:"100%", height:90, objectFit:"cover", display:"block" }}/>
              <div style={{ position:"absolute", bottom:0, left:0, right:0, background:"linear-gradient(transparent,rgba(0,0,0,.8))", padding:"18px 8px 6px" }}>
                <div style={{ fontSize:10, color:"#fff", fontWeight:700, fontFamily:"'Space Mono',monospace" }}>{img.name}</div>
              </div>
              <div style={{ position:"absolute", top:6, right:6, background:"rgba(0,0,0,.7)", borderRadius:6, padding:"2px 7px", fontSize:10, fontWeight:700, color:img.count>400?C.danger:C.accent, fontFamily:"monospace" }}>{img.count}</div>
            </div>
            <div style={{ padding:"6px 8px", background:dark?"#0d1117":"#f8faff" }}>
              <div style={{ fontSize:10, color:muted }}>{img.label}</div>
            </div>
          </div>
        ))}
      </div>
      <div style={{ padding:"10px 16px", borderTop:`1px solid ${bdr}`, fontSize:11, color:muted, display:"flex", alignItems:"center", gap:6 }}>
        <span style={{ color:C.accent }}>💡</span>
        You can also drag these images into the upload zone above, or replace the URLs with your local dataset paths.
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  THRESHOLD CONTROL
// ─────────────────────────────────────────────────────────────────────────────
function ThresholdControl({ value, onChange, dark }) {
  const bg=dark?"#161b27":"#fff", bdr=dark?"#2a3040":"#e2e8f8", muted=dark?"#5a6478":"#94a3b8", text=dark?"#e0e8ff":"#1a1f35";
  return (
    <div style={{ background:bg, border:`1px solid ${bdr}`, borderRadius:14, padding:"18px 20px" }}>
      <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:14 }}>
        <div style={{ width:30, height:30, borderRadius:8, background:C.warn+"22", display:"flex", alignItems:"center", justifyContent:"center" }}>
          <Shield size={15} color={C.warn}/>
        </div>
        <div style={{ flex:1 }}>
          <div style={{ fontSize:13, fontWeight:700, color:text }}>Alert Threshold</div>
          <div style={{ fontSize:11, color:muted }}>Trigger overcrowding alert above this count</div>
        </div>
        <div style={{ background:C.warn+"20", border:`1px solid ${C.warn}40`, borderRadius:8, padding:"4px 12px" }}>
          <span style={{ fontSize:18, fontWeight:800, fontFamily:"'Space Mono',monospace", color:C.warn }}>{value}</span>
        </div>
      </div>
      <input type="range" min={50} max={2000} step={10} value={value}
        onChange={e=>onChange(Number(e.target.value))}
        style={{ width:"100%", accentColor:C.warn, cursor:"pointer", marginBottom:8 }}/>
      <div style={{ display:"flex", justifyContent:"space-between", fontSize:10, color:muted, marginBottom:12 }}>
        <span>50</span><span>500</span><span>1000</span><span>2000</span>
      </div>
      <div style={{ display:"flex", gap:7, flexWrap:"wrap" }}>
        {[{l:"50",v:50},{l:"100",v:100},{l:"200",v:200},{l:"300",v:300},{l:"500",v:500},{l:"1000",v:1000}].map(({l,v})=>(
          <button key={v} onClick={()=>onChange(v)}
            style={{ background:value===v?C.warn+"30":dark?"#1a2030":"#f0f4ff", border:`1px solid ${value===v?C.warn:bdr}`, borderRadius:7, padding:"4px 10px", fontSize:11, fontWeight:600, color:value===v?C.warn:muted, cursor:"pointer", fontFamily:"inherit", transition:"all 0.15s" }}>
            {l}
          </button>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  TOAST
// ─────────────────────────────────────────────────────────────────────────────
function Toast({ msg, type, onClose }) {
  useEffect(()=>{ const t=setTimeout(onClose,4500); return ()=>clearTimeout(t); },[onClose]);
  const color = type==="success"?C.success:type==="alert"||type==="error"?C.danger:C.warn;
  const Ico   = type==="success"?CheckCircle:AlertTriangle;
  return (
    <div style={{ position:"fixed", top:24, right:24, zIndex:9999, background:color+"1a", border:`1px solid ${color}`, borderRadius:12, padding:"13px 18px", display:"flex", alignItems:"center", gap:10, boxShadow:`0 8px 32px ${color}33`, animation:"slideIn 0.3s ease", minWidth:300, maxWidth:400 }}>
      <Ico size={17} color={color}/>
      <span style={{ color:"#e0e8ff", fontSize:13, fontFamily:"inherit", flex:1 }}>{msg}</span>
      <button onClick={onClose} style={{ background:"none", border:"none", color:"#aaa", cursor:"pointer" }}><X size={13}/></button>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  STAT CARD
// ─────────────────────────────────────────────────────────────────────────────
function StatCard({ icon:Ico, label, value, sub, color, dark }) {
  return (
    <div
      onMouseEnter={e=>{e.currentTarget.style.transform="translateY(-2px)";e.currentTarget.style.boxShadow=`0 10px 32px ${color}22`;}}
      onMouseLeave={e=>{e.currentTarget.style.transform="";e.currentTarget.style.boxShadow="";}}
      style={{ background:dark?"#1a1f2e":"#fff", border:`1px solid ${dark?"#2a3040":"#e8ecf4"}`, borderRadius:14, padding:"16px 20px", display:"flex", flexDirection:"column", gap:7, position:"relative", overflow:"hidden", transition:"transform 0.2s,box-shadow 0.2s" }}>
      <div style={{ position:"absolute", top:-18, right:-18, width:72, height:72, borderRadius:"50%", background:color+"10" }}/>
      <div style={{ width:34, height:34, borderRadius:9, background:color+"20", display:"flex", alignItems:"center", justifyContent:"center" }}><Ico size={17} color={color}/></div>
      <div style={{ fontSize:24, fontWeight:800, color:dark?"#fff":"#111", fontFamily:"'Space Mono',monospace" }}>{value}</div>
      <div style={{ fontSize:12, color:dark?"#667":"#888", fontWeight:500 }}>{label}</div>
      {sub && <div style={{ fontSize:11, color, fontWeight:600 }}>{sub}</div>}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  MAIN APP
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [dark,           setDark]          = useState(true);
  const [page,           setPage]          = useState("upload");
  const [file,           setFile]          = useState(null);
  const [preview,        setPreview]       = useState(null);
  const [fileType,       setFileType]      = useState("image");
  const [progress,       setProgress]      = useState(0);
  const [processing,     setProcessing]    = useState(false);
  const [result,         setResult]        = useState(null);
  const [history,        setHistory]       = useState([]);
  const [toast,          setToast]         = useState(null);
  const [threshold,      setThreshold]     = useState(50);
  const [alertDismissed, setAlertDismissed]= useState(false);
  const [drag,           setDrag]          = useState(false);
  const [status,         setStatus]        = useState("Active");
  const [selectedSample, setSelectedSample]= useState(null);
  const inputRef = useRef();

  const notify = useCallback((msg,type="success")=>setToast({msg,type}),[]);
  useEffect(()=>{ API.getHistory().then(setHistory); },[]);

  const totalDetected = history.reduce((a,b)=>a+b.count,0);
  const avgCount      = history.length ? Math.floor(totalDetected/history.length) : 0;
  const alertCount    = history.filter(h=>h.count>=threshold).length;
  const chartData     = history.slice(0,8).map((h,i)=>({ time:`#${i+1}`, count:h.count, threshold }));

  const progressMsg = progress<20?"Uploading…":progress<45?"Running sliding-window patch inference…":progress<70?"Calibrating scale factor…":progress<90?"Generating density map…":"Finalising…";

  // handle real file drop / pick
  const handleFile = (f) => {
    if (!f) return;
    if (!["image/jpeg","image/png","video/mp4"].includes(f.type)) { notify("Only JPG, PNG, MP4 supported.","error"); return; }
    setFile(f); setPreview(URL.createObjectURL(f));
    setFileType(f.type.startsWith("video")?"video":"image");
    setResult(null); setProgress(0); setAlertDismissed(false); setSelectedSample(null);
    notify(`"${f.name}" ready — click Analyse`,"success");
  };

  // handle sample image selection
  const handleSampleSelect = (img) => {
    setSelectedSample(img);
    setPreview(img.url);
    setFileType("image");
    setFile({ name: img.name, type:"image/jpeg", size:0 });
    setResult(null); setProgress(0); setAlertDismissed(false);
    notify(`Sample "${img.name}" loaded — click Analyse`,"success");
  };

  const handleAnalyse = async () => {
    if (!file && !selectedSample) { notify("Drop a file or pick a sample image first.","error"); return; }
    setProcessing(true); setStatus("Processing"); setProgress(0); setAlertDismissed(false);
    try {
      const res = await API.analyse(file, threshold, setProgress, selectedSample?.count);
      setResult(res);
      setHistory(h=>[{ id:Date.now(), file:file?.name||selectedSample?.name, count:res.count, time:res.time, date:"Today", threshold, alert:res.alert }, ...h]);
      if (res.alert) notify(`⚠ Overcrowding! ${res.count} people — threshold is ${threshold}.`,"alert");
      else           notify(`Done! ${res.count} people detected — within threshold.`,"success");
    } catch(e) { notify("Processing failed. Check backend.","error"); }
    finally { setProcessing(false); setStatus("Active"); }
  };

  const bg   = dark?"#0d1117":"#f0f4ff";
  const card = dark?"#161b27":"#ffffff";
  const bdr  = dark?"#1e2535":"#e2e8f8";
  const text = dark?"#e0e8ff":"#1a1f35";
  const muted= dark?"#5a6478":"#94a3b8";

  const navItems=[
    {id:"upload",   icon:Upload,   label:"Analyse"},
    {id:"dashboard",icon:Home,     label:"Dashboard"},
    {id:"analytics",icon:BarChart3,label:"Analytics"},
  ];

  return (
    <div style={{ minHeight:"100vh", background:bg, fontFamily:"'Syne',sans-serif", display:"flex", color:text }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:4px} ::-webkit-scrollbar-thumb{background:#2a3040;border-radius:99px}
        input[type=range]{height:4px;width:100%}
        @keyframes slideIn{from{transform:translateX(100px);opacity:0}to{transform:none;opacity:1}}
        @keyframes slideDown{from{transform:translateY(-16px);opacity:0}to{transform:none;opacity:1}}
        @keyframes fadeUp{from{transform:translateY(18px);opacity:0}to{transform:none;opacity:1}}
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
        @keyframes alertPulse{0%,100%{opacity:.65;box-shadow:0 0 0 0 #ff475744}50%{opacity:1;box-shadow:0 0 0 7px #ff475712}}
        @keyframes alertBorder{0%,100%{opacity:.6}50%{opacity:1}}
      `}</style>

      {/* ── SIDEBAR ── */}
      <div style={{ width:210, background:dark?"#0a0e1a":"#fff", borderRight:`1px solid ${bdr}`, display:"flex", flexDirection:"column", padding:"22px 0", position:"sticky", top:0, height:"100vh", flexShrink:0 }}>

        <div style={{ padding:"0 18px 18px", borderBottom:`1px solid ${bdr}` }}>
          <div style={{ display:"flex", alignItems:"center", gap:10 }}>
            <div style={{ width:34, height:34, borderRadius:10, background:`linear-gradient(135deg,${C.accent},${C.accent2})`, display:"flex", alignItems:"center", justifyContent:"center" }}><Users size={17} color="#fff"/></div>
            <div><div style={{ fontSize:13, fontWeight:800, letterSpacing:-0.5 }}>CrowdVision</div><div style={{ fontSize:10, color:muted, letterSpacing:1, textTransform:"uppercase" }}>CSRNet · AI</div></div>
          </div>
        </div>

        {/* threshold pill */}
        <div style={{ padding:"10px 12px", borderBottom:`1px solid ${bdr}` }}>
          <div style={{ background:C.warn+"15", border:`1px solid ${C.warn}30`, borderRadius:9, padding:"8px 11px", display:"flex", alignItems:"center", gap:7 }}>
            <Bell size={12} color={C.warn}/>
            <div style={{ flex:1 }}>
              <div style={{ fontSize:9, color:C.warn, fontWeight:700, letterSpacing:"0.06em", textTransform:"uppercase" }}>Alert Threshold</div>
              <div style={{ fontSize:15, fontWeight:800, fontFamily:"'Space Mono',monospace", color:C.warn }}>{threshold}</div>
            </div>
            {result && (
              <div style={{ textAlign:"right" }}>
                <div style={{ fontSize:9, color:muted, textTransform:"uppercase", letterSpacing:"0.04em" }}>Current</div>
                <div style={{ fontSize:15, fontWeight:800, fontFamily:"'Space Mono',monospace", color:result.alert?C.danger:C.success }}>{result.count}</div>
              </div>
            )}
          </div>
        </div>

        <nav style={{ flex:1, padding:"12px 10px", display:"flex", flexDirection:"column", gap:3 }}>
          {navItems.map(({id,icon:Ico,label})=>{
            const active=page===id;
            return (
              <button key={id} onClick={()=>setPage(id)} style={{ display:"flex", alignItems:"center", gap:9, padding:"9px 11px", borderRadius:9, cursor:"pointer", background:active?C.accent+"18":"transparent", border:active?`1px solid ${C.accent}30`:"1px solid transparent", color:active?C.accent:muted, fontSize:13, fontWeight:active?700:500, fontFamily:"inherit", transition:"all 0.15s", textAlign:"left" }}>
                <Ico size={15}/>{label}
                {id==="upload"&&result&&result.alert&&<span style={{ marginLeft:"auto", width:7, height:7, borderRadius:"50%", background:C.danger, animation:"pulse 1s infinite" }}/>}
                {active&&<ChevronRight size={11} style={{ marginLeft:"auto" }}/>}
              </button>
            );
          })}
        </nav>

        <div style={{ padding:"12px 14px", borderTop:`1px solid ${bdr}` }}>
          <div style={{ display:"flex", alignItems:"center", gap:7, marginBottom:9 }}>
            <div style={{ width:7, height:7, borderRadius:"50%", background:status==="Active"?C.success:C.warn, boxShadow:`0 0 5px ${status==="Active"?C.success:C.warn}`, animation:status==="Processing"?"pulse 1s infinite":"none" }}/>
            <span style={{ fontSize:11, color:muted, fontWeight:600 }}>System {status}</span>
            {status==="Processing"&&<RefreshCw size={11} color={C.warn} style={{ animation:"spin 1s linear infinite", marginLeft:2 }}/>}
          </div>
          <button onClick={()=>setDark(!dark)} style={{ display:"flex", alignItems:"center", gap:7, background:dark?"#1a2030":"#eef2fb", border:"none", borderRadius:8, padding:"7px 11px", cursor:"pointer", color:text, fontSize:11, fontFamily:"inherit", fontWeight:600, width:"100%" }}>
            {dark?<Sun size={12}/>:<Moon size={12}/>}{dark?"Light Mode":"Dark Mode"}
          </button>
        </div>
      </div>

      {/* ── MAIN ── */}
      <div style={{ flex:1, overflow:"auto", padding:"26px 30px" }}>

        {/* ═══════════════════ ANALYSE PAGE ═══════════════════ */}
        {page==="upload" && (
          <div style={{ animation:"fadeUp 0.4s ease" }}>
            <div style={{ marginBottom:20 }}>
              <h1 style={{ fontSize:24, fontWeight:800, letterSpacing:-0.8 }}>Crowd Analysis</h1>
              <p style={{ fontSize:12, color:muted, marginTop:3 }}>Upload an image/video or pick a sample — get INPUT · OVERLAY · DENSITY MAP instantly</p>
            </div>

            <div style={{ display:"grid", gridTemplateColumns:"1fr 320px", gap:18, marginBottom:18 }}>

              {/* ─ drop zone ─ */}
              <div style={{ background:card, border:`1px solid ${bdr}`, borderRadius:16, padding:20 }}>
                <div
                  onClick={()=>inputRef.current.click()}
                  onDragOver={e=>{e.preventDefault();setDrag(true);}}
                  onDragLeave={()=>setDrag(false)}
                  onDrop={e=>{
                    e.preventDefault(); setDrag(false);
                    const sid=e.dataTransfer.getData("sampleId");
                    if(sid){ const s=SAMPLE_IMAGES.find(i=>i.id===parseInt(sid)); if(s)handleSampleSelect(s); return; }
                    handleFile(e.dataTransfer.files[0]);
                  }}
                  style={{ border:`2px dashed ${drag?C.accent:dark?"#2a3040":"#d0d8ea"}`, borderRadius:14, padding:"28px 20px", display:"flex", flexDirection:"column", alignItems:"center", gap:12, cursor:"pointer", transition:"all 0.2s", background:drag?C.accent+"08":dark?"#0a0e1a":"#f8faff", textAlign:"center", marginBottom:16 }}>
                  <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.mp4" style={{ display:"none" }} onChange={e=>handleFile(e.target.files[0])}/>
                  <div style={{ width:56, height:56, borderRadius:16, background:C.accent+"18", display:"flex", alignItems:"center", justifyContent:"center", border:`1px solid ${C.accent}30` }}><Upload size={24} color={C.accent}/></div>
                  <div>
                    <div style={{ fontSize:15, fontWeight:700, color:dark?"#e0e8ff":"#1a1f35", marginBottom:4 }}>Drop image / video here or click to browse</div>
                    <div style={{ fontSize:12, color:muted }}>JPG · PNG · MP4 — or drag sample images from below</div>
                  </div>
                  <div style={{ display:"flex", gap:8 }}>
                    {[{i:ImageIcon,l:"JPG/PNG"},{i:Play,l:"MP4 Video"}].map(({i:I,l})=>(
                      <div key={l} style={{ display:"flex", alignItems:"center", gap:5, background:dark?"#1a2030":"#eef2fb", borderRadius:7, padding:"4px 10px", fontSize:11, color:dark?"#8899bb":"#667" }}><I size={11}/>{l}</div>
                    ))}
                  </div>
                </div>

                {/* preview */}
                {preview && (
                  <div style={{ marginBottom:14 }}>
                    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:8 }}>
                      <span style={{ fontSize:12, fontWeight:600, color:muted }}>{selectedSample ? `Sample: ${selectedSample.name}` : `File: ${file?.name}`}</span>
                      <button onClick={()=>{setFile(null);setPreview(null);setResult(null);setSelectedSample(null);}} style={{ background:"none", border:"none", color:muted, cursor:"pointer", fontSize:12 }}>Remove ×</button>
                    </div>
                    <div style={{ borderRadius:10, overflow:"hidden", border:`1px solid ${bdr}`, maxHeight:200 }}>
                      {fileType==="video"
                        ? <video src={preview} controls style={{ width:"100%", maxHeight:200, objectFit:"cover", display:"block" }}/>
                        : <img src={preview} alt="preview" style={{ width:"100%", maxHeight:200, objectFit:"cover", display:"block" }}/>
                      }
                    </div>
                  </div>
                )}

                {/* progress */}
                {processing && (
                  <div style={{ marginBottom:14 }}>
                    <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
                      <span style={{ fontSize:11, color:muted }}>{progressMsg}</span>
                      <span style={{ fontSize:11, color:C.accent, fontFamily:"monospace", fontWeight:700 }}>{progress.toFixed(0)}%</span>
                    </div>
                    <div style={{ height:5, borderRadius:99, background:dark?"#2a3040":"#e8ecf4", overflow:"hidden" }}>
                      <div style={{ height:"100%", borderRadius:99, width:`${progress}%`, transition:"width 0.3s ease", background:`linear-gradient(90deg,${C.accent},${C.accent2})` }}/>
                    </div>
                    <div style={{ display:"flex", justifyContent:"center", marginTop:12 }}>
                      <div style={{ width:22, height:22, borderRadius:"50%", border:`2.5px solid ${C.accent}30`, borderTop:`2.5px solid ${C.accent}`, animation:"spin 0.8s linear infinite" }}/>
                    </div>
                  </div>
                )}

                <button onClick={handleAnalyse} disabled={(!file&&!selectedSample)||processing} style={{ width:"100%", background:((file||selectedSample)&&!processing)?`linear-gradient(135deg,${C.accent},${C.accent2})`:dark?"#1a2030":"#e8ecf4", border:"none", borderRadius:11, padding:"13px 0", fontSize:14, fontWeight:800, color:((file||selectedSample)&&!processing)?"#0a0e1a":muted, cursor:((file||selectedSample)&&!processing)?"pointer":"not-allowed", fontFamily:"inherit", transition:"all 0.2s", boxShadow:((file||selectedSample)&&!processing)?`0 7px 22px ${C.accent}40`:"none" }}>
                  {processing ? "Analysing…" : "▶  Analyse Crowd"}
                </button>
              </div>

              {/* ─ threshold control ─ */}
              <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
                <ThresholdControl value={threshold} onChange={v=>{setThreshold(v);setAlertDismissed(false);}} dark={dark}/>
                <div style={{ background:card, border:`1px solid ${bdr}`, borderRadius:14, padding:"16px 18px" }}>
                  <div style={{ fontSize:12, fontWeight:700, color:text, marginBottom:12 }}>Output panels</div>
                  {[
                    {color:C.accent, label:"INPUT",          desc:"Original image or video frame"},
                    {color:C.accent2,label:"OVERLAY DENSITY",desc:"Crowd density overlay on input"},
                    {color:C.warn,   label:"DENSITY HEATMAP",desc:"Raw density map — hot = dense"},
                  ].map(({color,label,desc})=>(
                    <div key={label} style={{ display:"flex", alignItems:"flex-start", gap:8, marginBottom:10 }}>
                      <div style={{ width:7, height:7, borderRadius:"50%", background:color, marginTop:4, flexShrink:0 }}/>
                      <div><div style={{ fontSize:11, fontWeight:700, color:text }}>{label}</div><div style={{ fontSize:10, color:muted }}>{desc}</div></div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* ─ RESULT OUTPUT ─ */}
            {(result || preview) && (
              <div style={{ background:card, border:`1px solid ${bdr}`, borderRadius:16, padding:20, marginBottom:18 }}>
                {/* stats row */}
                {result && <StatsRow result={result} dark={dark}/>}

                {/* alert banner */}
                {result?.alert && !alertDismissed && (
                  <AlertBanner count={result.count} threshold={threshold} onDismiss={()=>setAlertDismissed(true)}/>
                )}

                {/* three panels */}
                <ThreePanelOutput result={result} preview={preview} fileType={fileType} threshold={threshold} dark={dark}/>

                {/* crowd count live bar */}
                {result && <CrowdCountBar result={result} threshold={threshold} dark={dark}/>}

                {/* download row */}
                {result && (
                  <div style={{ display:"flex", gap:10, marginTop:14 }}>
                    {[{l:"Download Input",c:C.accent},{l:"Download Heatmap",c:C.accent2},{l:"Export Report",c:C.warn}].map(({l,c})=>(
                      <button key={l} style={{ display:"flex", alignItems:"center", gap:6, background:c+"18", border:`1px solid ${c}30`, borderRadius:9, padding:"7px 14px", color:c, cursor:"pointer", fontSize:12, fontFamily:"inherit", fontWeight:600 }}>
                        <Download size={13}/>{l}
                      </button>
                    ))}
                    <button onClick={()=>{setResult(null);setProgress(0);}} style={{ display:"flex", alignItems:"center", gap:6, background:dark?"#1a2030":"#f0f4ff", border:`1px solid ${bdr}`, borderRadius:9, padding:"7px 14px", color:muted, cursor:"pointer", fontSize:12, fontFamily:"inherit", fontWeight:600, marginLeft:"auto" }}>
                      <RefreshCw size={13}/> Reset
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* ─ SAMPLE IMAGES ─ */}
            <SamplePanel onSelect={handleSampleSelect} dark={dark}/>
          </div>
        )}

        {/* ═══════════════════ DASHBOARD ═══════════════════ */}
        {page==="dashboard" && (
          <div style={{ animation:"fadeUp 0.4s ease" }}>
            <div style={{ marginBottom:22 }}>
              <h1 style={{ fontSize:24, fontWeight:800, letterSpacing:-0.8 }}>Dashboard</h1>
              <p style={{ fontSize:12, color:muted, marginTop:3 }}>Session overview — alert threshold: <strong style={{ color:C.warn }}>{threshold}</strong></p>
            </div>

            <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:13, marginBottom:18 }}>
              <StatCard icon={Users}         label="Total Detected"  value={totalDetected.toLocaleString()} sub="All sessions"       color={C.accent}  dark={dark}/>
              <StatCard icon={Activity}      label="Avg Per Session" value={avgCount}                        sub="Per analysis"        color={C.blue}    dark={dark}/>
              <StatCard icon={AlertTriangle} label="Alerts"          value={alertCount}                      sub={`Above ${threshold}`}color={C.danger}  dark={dark}/>
              <StatCard icon={Zap}           label="Status"          value={status}                          sub="Live system"         color={C.success} dark={dark}/>
            </div>

            {result && (
              <div style={{ background:card, border:`1px solid ${bdr}`, borderRadius:16, padding:20, marginBottom:16 }}>
                <div style={{ fontSize:13, fontWeight:700, marginBottom:14 }}>Last Analysis Result</div>
                {result.alert && !alertDismissed && <AlertBanner count={result.count} threshold={threshold} onDismiss={()=>setAlertDismissed(true)}/>}
                <StatsRow result={result} dark={dark}/>
                <ThreePanelOutput result={result} preview={preview} fileType={fileType} threshold={threshold} dark={dark}/>
                <CrowdCountBar result={result} threshold={threshold} dark={dark}/>
              </div>
            )}

            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
              <div style={{ background:card, border:`1px solid ${bdr}`, borderRadius:14, padding:20 }}>
                <div style={{ fontSize:13, fontWeight:700, marginBottom:16 }}>Count vs Threshold Trend</div>
                <ResponsiveContainer width="100%" height={180}>
                  <AreaChart data={chartData}>
                    <defs><linearGradient id="cg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.accent} stopOpacity={0.3}/><stop offset="95%" stopColor={C.accent} stopOpacity={0}/></linearGradient></defs>
                    <CartesianGrid strokeDasharray="3 3" stroke={dark?"#1e2535":"#eee"}/>
                    <XAxis dataKey="time" tick={{ fill:muted, fontSize:10 }} axisLine={false}/>
                    <YAxis tick={{ fill:muted, fontSize:10 }} axisLine={false}/>
                    <Tooltip contentStyle={{ background:dark?"#1a2030":"#fff", border:`1px solid ${bdr}`, borderRadius:8, color:text }}/>
                    <Area  type="monotone" dataKey="count"     stroke={C.accent} fill="url(#cg)" strokeWidth={2} dot={false} name="Count"/>
                    <Line  type="monotone" dataKey="threshold" stroke={C.danger} strokeWidth={1.5} strokeDasharray="5 4" dot={false} name="Threshold"/>
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <ThresholdControl value={threshold} onChange={v=>{setThreshold(v);setAlertDismissed(false);}} dark={dark}/>
            </div>
          </div>
        )}

        {/* ═══════════════════ ANALYTICS ═══════════════════ */}
        {page==="analytics" && (
          <div style={{ animation:"fadeUp 0.4s ease" }}>
            <div style={{ marginBottom:22 }}>
              <h1 style={{ fontSize:24, fontWeight:800, letterSpacing:-0.8 }}>Analytics</h1>
              <p style={{ fontSize:12, color:muted, marginTop:3 }}>Full session history · alert rate · trend analysis</p>
            </div>

            <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:13, marginBottom:18 }}>
              <StatCard icon={TrendingUp}    label="Peak Count"  value={Math.max(...history.map(h=>h.count),0).toLocaleString()} sub="All sessions"         color={C.danger}  dark={dark}/>
              <StatCard icon={Users}         label="Average"     value={avgCount}                                                  sub="Per session"          color={C.accent}  dark={dark}/>
              <StatCard icon={AlertTriangle} label="Alert Rate"  value={history.length?Math.round(alertCount/history.length*100)+"%":"0%"} sub={`Threshold: ${threshold}`} color={C.warn} dark={dark}/>
              <StatCard icon={Database}      label="Sessions"    value={history.length}                                            sub="Total processed"      color={C.blue}    dark={dark}/>
            </div>

            <div style={{ background:card, border:`1px solid ${bdr}`, borderRadius:14, padding:20, marginBottom:16 }}>
              <div style={{ fontSize:13, fontWeight:700, marginBottom:14 }}>Count vs Alert Threshold</div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={dark?"#1e2535":"#eee"}/>
                  <XAxis dataKey="time" tick={{ fill:muted, fontSize:10 }} axisLine={false}/>
                  <YAxis tick={{ fill:muted, fontSize:10 }} axisLine={false}/>
                  <Tooltip contentStyle={{ background:dark?"#1a2030":"#fff", border:`1px solid ${bdr}`, borderRadius:8, color:text }}/>
                  <Line type="monotone" dataKey="count"     stroke={C.accent} strokeWidth={2.5} dot={{ fill:C.accent, r:3 }} name="Count"/>
                  <Line type="monotone" dataKey="threshold" stroke={C.danger} strokeWidth={1.5} strokeDasharray="5 4" dot={false} name="Threshold"/>
                </LineChart>
              </ResponsiveContainer>
              <div style={{ display:"flex", gap:16, marginTop:8, fontSize:11, color:muted }}>
                <span style={{ display:"flex", alignItems:"center", gap:5 }}><span style={{ width:18, height:2.5, background:C.accent, borderRadius:2, display:"inline-block" }}/> Count</span>
                <span style={{ display:"flex", alignItems:"center", gap:5 }}><span style={{ width:18, height:2, background:C.danger, borderRadius:2, display:"inline-block", opacity:.8 }}/> Threshold ({threshold})</span>
              </div>
            </div>

            <div style={{ background:card, border:`1px solid ${bdr}`, borderRadius:14, overflow:"hidden" }}>
              <div style={{ padding:"13px 18px", borderBottom:`1px solid ${bdr}`, fontSize:13, fontWeight:700 }}>Full Session History</div>
              <div style={{ overflowX:"auto" }}>
                <table style={{ width:"100%", borderCollapse:"collapse" }}>
                  <thead><tr style={{ background:dark?"#0d1117":"#f8faff" }}>
                    {["#","File","Date","Time","Count","Threshold","Status"].map(h=><th key={h} style={{ textAlign:"left", fontSize:10, fontWeight:700, color:muted, padding:"8px 14px", textTransform:"uppercase", letterSpacing:1 }}>{h}</th>)}
                  </tr></thead>
                  <tbody>
                    {history.map((row,i)=>(
                      <tr key={row.id} onMouseEnter={e=>e.currentTarget.style.background=dark?"#1a2030":"#f8faff"} onMouseLeave={e=>e.currentTarget.style.background=""} style={{ borderTop:`1px solid ${bdr}`, transition:"background 0.12s" }}>
                        <td style={{ padding:"10px 14px", fontSize:10, color:muted, fontFamily:"monospace" }}>{String(i+1).padStart(3,"0")}</td>
                        <td style={{ padding:"10px 14px", fontSize:12, fontWeight:600 }}>{row.file}</td>
                        <td style={{ padding:"10px 14px", fontSize:11, color:muted }}>{row.date}</td>
                        <td style={{ padding:"10px 14px", fontSize:11, color:muted, fontFamily:"monospace" }}>{row.time}</td>
                        <td style={{ padding:"10px 14px" }}><span style={{ fontSize:13, fontWeight:800, fontFamily:"monospace", color:row.count>=(row.threshold||threshold)?C.danger:row.count>(row.threshold||threshold)*0.7?C.warn:C.accent }}>{row.count}</span></td>
                        <td style={{ padding:"10px 14px", fontSize:11, color:muted, fontFamily:"monospace" }}>{row.threshold||threshold}</td>
                        <td style={{ padding:"10px 14px" }}><span style={{ fontSize:10, fontWeight:700, padding:"2px 9px", borderRadius:99, background:row.alert?C.danger+"20":C.success+"20", color:row.alert?C.danger:C.success }}>{row.alert?"ALERT":"NORMAL"}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>

      {toast && <Toast msg={toast.msg} type={toast.type} onClose={()=>setToast(null)}/>}
    </div>
  );
}